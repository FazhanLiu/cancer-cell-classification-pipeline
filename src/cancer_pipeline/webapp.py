"""Single-user, loopback-only case workbench. Not a multi-tenant clinical service."""
import atexit
import io
import logging
import os
import re
import secrets
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from cancer_pipeline.artifacts import read_json, save_json
from cancer_pipeline.case_reports import generate_case_report
from cancer_pipeline.config import load_environment
from cancer_pipeline.inference import PatchInference
from cancer_pipeline.reports import _safe_error

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
CASE_ID = re.compile(r'^[a-f0-9]{32}$')
IMAGE_ID = re.compile(r'^image_[0-9]{3}$')
BUSY = {'queued', 'running'}


def now():
    return datetime.now(timezone.utc).isoformat()


class CaseStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        for path in self.root.glob('*/case.json'):
            case = read_json(path)
            if case.get('status') in BUSY:
                case['status'] = 'interrupted'
                case['message'] = '上次运行被中断，点击继续分析即可保留已完成结果并重试。'
                for image in case['images']:
                    if image['status'] in {'classifying', 'explaining'}:
                        image['status'] = 'interrupted'
                if case['report']['status'] == 'generating':
                    case['report'] = {'status': 'failed', 'error': '报告生成期间服务中断，可重试。'}
                save_json(case, path)

    def directory(self, case_id):
        if not CASE_ID.fullmatch(case_id):
            raise KeyError(case_id)
        return self.root / case_id

    def get(self, case_id):
        with self.lock:
            path = self.directory(case_id) / 'case.json'
            if not path.is_file():
                raise KeyError(case_id)
            return read_json(path)

    def update(self, case_id, operation):
        with self.lock:
            case = self.get(case_id)
            operation(case)
            case['updated_utc'] = now()
            save_json(case, self.directory(case_id) / 'case.json')
            return case

    def summaries(self):
        with self.lock:
            rows = [read_json(p) for p in self.root.glob('*/case.json')]
        return [{'id': c['id'], 'reference': c['reference'], 'status': c['status'],
                 'created_utc': c['created_utc'], 'image_count': len(c['images'])}
                for c in sorted(rows, key=lambda x: x['created_utc'], reverse=True)[:100]]

    def create(self, reference, images):
        case_id = uuid.uuid4().hex
        directory = self.directory(case_id)
        directory.mkdir(exist_ok=False)
        items = []
        try:
            for index, image in enumerate(images, 1):
                image_id = f'image_{index:03d}'
                folder = directory / image_id
                folder.mkdir()
                image['image'].save(folder / 'input.png')
                preview = image['image'].copy()
                preview.thumbnail((720, 720))
                preview.save(folder / 'preview.png')
                preview.close()
                items.append({'id': image_id, 'filename': image['filename'], 'width': image['image'].width,
                              'height': image['image'].height, 'status': 'uploaded',
                              'prediction': None, 'evidence': None, 'error': None})
                image['image'].close()
            case = {'schema_version': 1, 'id': case_id, 'reference': reference or f'CASE-{case_id[:8].upper()}',
                    'created_utc': now(), 'updated_utc': now(), 'status': 'uploaded',
                    'message': '图片已保存，等待开始分析。', 'images': items,
                    'model': None, 'report': {'status': 'not_started'}}
            save_json(case, directory / 'case.json')
            return case
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise


class CaseJobs:
    def __init__(self, store, runtime, llm_model, llm_caller=None):
        self.store, self.runtime = store, runtime
        self.llm_model, self.llm_caller = llm_model, llm_caller
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='patch-analysis')
        self.lock = threading.Lock()
        self.active = set()

    def submit(self, case_id):
        if self.runtime is None:
            raise ValueError('尚未加载训练模型，请在 .env 配置 CANCER_MODEL_PATH 后重启服务。')
        with self.lock:
            case = self.store.get(case_id)
            if case_id in self.active:
                raise ValueError('此病例正在分析，请等待当前任务完成。')
            if len(self.active) >= 8:
                raise ValueError('本机待处理任务已满，请稍后重试。')
            if case['status'] == 'complete':
                raise ValueError('此病例已完成，无需重复调用模型或 DeepSeek。')
            if case.get('model') and case['model'].get('analysis_hash') != self.runtime.info().get('analysis_hash'):
                raise ValueError('模型、阈值或解释参数已变更，请新建病例，避免混合不同设置的结果。')
            case = self.store.update(case_id, lambda c: c.update(status='queued', message='已加入本机分析队列。'))
            self.active.add(case_id)
            self.executor.submit(self._run, case_id)
        return case

    def _run(self, case_id):
        try:
            self.store.update(case_id, lambda c: c.update(status='running', model=self.runtime.info()))
            snapshot = self.store.get(case_id)
            for index, item in enumerate(snapshot['images'], 1):
                if item['status'] == 'complete':
                    continue
                image_id = item['id']

                def progress(stage, prediction, image_id=image_id, index=index):
                    def update(case):
                        target = next(x for x in case['images'] if x['id'] == image_id)
                        target.update(status=stage, error=None)
                        if prediction:
                            target['prediction'] = prediction
                        name = '模型分类' if stage == 'classifying' else '计算三种解释与融合图'
                        case['message'] = f"图像 {index}/{len(case['images'])}：{name}"
                    self.store.update(case_id, update)

                try:
                    folder = self.store.directory(case_id) / image_id
                    prediction, evidence = self.runtime.analyze(folder / 'input.png', folder, image_id, progress)
                    def finish(case):
                        target = next(x for x in case['images'] if x['id'] == image_id)
                        target.update(status='complete', prediction=prediction, evidence=evidence, error=None)
                    self.store.update(case_id, finish)
                except Exception as error:
                    message = _safe_error(error)
                    def fail(case, message=message):
                        target = next(x for x in case['images'] if x['id'] == image_id)
                        target.update(status='failed', error=message)
                    self.store.update(case_id, fail)
            self.store.update(case_id, lambda c: c.update(message='整理逐图证据，生成中文病例汇总。', report={'status': 'generating'}))
            snapshot = self.store.get(case_id)
            if not any(x.get('prediction') for x in snapshot['images']):
                report = {'status': 'failed', 'error': '所有图片分类失败，未调用 DeepSeek，请检查图像与模型。'}
            elif not os.environ.get('DEEPSEEK_API_KEY', '').strip():
                report = {'status': 'failed', 'error': '尚未配置 DeepSeek Key。请填写 .env 并重启服务，再继续分析。'}
            else:
                report = generate_case_report(snapshot, self.store.directory(case_id), self.llm_model, caller=self.llm_caller)
            def complete(case):
                case['report'] = report
                all_images = all(x['status'] == 'complete' for x in case['images'])
                classified = any(x.get('prediction') for x in case['images'])
                case['status'] = 'complete' if all_images and report['status'] == 'complete' else ('partial' if classified else 'failed')
                case['message'] = '逐图分析与病例报告已完成，请复核结果。' if case['status'] == 'complete' else '部分步骤未完成，已保存成功结果，可继续分析重试。'
            self.store.update(case_id, complete)
        except Exception as error:
            message = _safe_error(error)
            LOGGER.error('Case task failed: %s', message)
            def fail_case(case):
                case.update(status='failed', message=message)
                if case['report']['status'] == 'generating':
                    case['report'] = {'status': 'failed', 'error': message}
            self.store.update(case_id, fail_case)
        finally:
            with self.lock:
                self.active.discard(case_id)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def read_upload(storage):
    name = (storage.filename or '').replace('\\', '/').split('/')[-1][:180]
    if Path(name).suffix.lower() not in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'}:
        raise ValueError('仅支持 PNG、JPG、TIFF、WebP 病理小图。')
    data = storage.stream.read(10 * 1024 * 1024 + 1)
    if not data or len(data) > 10 * 1024 * 1024:
        raise ValueError('每张图片必须非空且不超过 10 MB。')
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {'PNG', 'JPEG', 'TIFF', 'WEBP'}:
                raise ValueError('图片内容格式不受支持。')
            if getattr(source, 'n_frames', 1) != 1:
                raise ValueError('仅支持单页病理小图；多页 TIFF 或完整切片需要单独的切片处理流程。')
            if source.width * source.height > 16_777_216 or min(source.size) < 16:
                raise ValueError('图片尺寸需至少 16×16，总像素不超过 4096×4096。请上传已裁剪的 patch。')
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('无法读取图片，请检查文件是否损坏。') from error
    return {'filename': name, 'image': image}


def create_app(settings=None, runtime=None, llm_caller=None):
    load_environment(ROOT / '.env')
    app = Flask(__name__, template_folder='web/templates', static_folder='web/assets', static_url_path='/assets')
    app.config.update(OUTPUT_ROOT=str(ROOT / 'outputs' / 'web_cases'),
                      MODEL_PATH=os.environ.get('CANCER_MODEL_PATH', ''),
                      DECISION_PATH=os.environ.get('CANCER_DECISION_PATH', ''),
                      DEVICE=os.environ.get('CANCER_DEVICE', 'auto'),
                      LLM_MODEL=os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash'),
                      MAX_CONTENT_LENGTH=100 * 1024 * 1024,
                      MAX_FORM_MEMORY_SIZE=32 * 1024, MAX_FORM_PARTS=24,
                      TRUSTED_HOSTS=['localhost', '127.0.0.1', '[::1]'],
                      PROXY_TOKEN=None,
                      DEPLOYMENT_ENV='local',
                      RELEASE='local',
                      LOCAL_TOKEN=secrets.token_urlsafe(32))
    app.config.update(settings or {})
    model_error = None
    if runtime is None and app.config['MODEL_PATH']:
        try:
            checkpoint = Path(app.config['MODEL_PATH']).expanduser()
            decision = Path(app.config['DECISION_PATH']).expanduser() if app.config['DECISION_PATH'] else None
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
            if decision is not None and not decision.is_absolute():
                decision = ROOT / decision
            runtime = PatchInference(checkpoint, decision, app.config['DEVICE'])
        except Exception as error:
            model_error = _safe_error(error)
    store = CaseStore(app.config['OUTPUT_ROOT'])
    jobs = CaseJobs(store, runtime, app.config['LLM_MODEL'], llm_caller)
    app.extensions.update(case_store=store, case_jobs=jobs)
    atexit.register(jobs.close)

    @app.before_request
    def local_boundary():
        if request.remote_addr not in {None, '127.0.0.1', '::1'}:
            abort(403, description='此版本仅支持本机访问。')
        # Each hosted environment trusts only its own reverse proxy credential.
        # Even another process on this server cannot read case data through HTTP.
        if request.path != '/healthz' and app.config['PROXY_TOKEN']:
            if not secrets.compare_digest(request.headers.get('X-Cancer-Proxy-Token', ''),
                                          app.config['PROXY_TOKEN']):
                abort(403)
        if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
            if not secrets.compare_digest(request.headers.get('X-Local-Token', ''), app.config['LOCAL_TOKEN']):
                abort(403, description='请求已过期，请刷新页面后重试。')

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        message = '上传总量超过 100 MB，请减少图片数量或大小。' if error.code == 413 else error.description
        return jsonify(error=message), error.code

    @app.errorhandler(KeyError)
    def missing(_error):
        return jsonify(error='未找到对应病例或图片。'), 404

    @app.get('/')
    def index():
        return render_template('index.html', token=app.config['LOCAL_TOKEN'])

    @app.get('/healthz')
    def health():
        healthy = not app.config['MODEL_PATH'] or runtime is not None
        return jsonify(status='ok' if healthy else 'unhealthy',
                       environment=app.config['DEPLOYMENT_ENV'], release=app.config['RELEASE'],
                       model_ready=runtime is not None), 200 if healthy else 503

    @app.get('/api/status')
    def status():
        return jsonify(model_ready=runtime is not None, model=runtime.info() if runtime else None,
                       model_error=model_error, llm_ready=bool(os.environ.get('DEEPSEEK_API_KEY', '').strip()),
                       llm_model=app.config['LLM_MODEL'], max_images=20, max_file_mb=10,
                       local_only=True)

    @app.get('/api/cases')
    def list_cases():
        return jsonify(cases=store.summaries())

    @app.post('/api/cases')
    def upload_case():
        files = request.files.getlist('images')
        if not 1 <= len(files) <= 20:
            return jsonify(error='每个病例请上传 1–20 张病理小图。'), 400
        reference = request.form.get('reference', '').strip()
        if len(reference) > 80 or '\n' in reference or '\r' in reference:
            return jsonify(error='病例备注请使用不超过 80 字符的单行文本。'), 400
        try:
            case = store.create(reference, (read_upload(f) for f in files))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        return jsonify(case), 201

    @app.get('/api/cases/<case_id>')
    def get_case(case_id):
        return jsonify(store.get(case_id))

    @app.post('/api/cases/<case_id>/analyze')
    def analyze_case(case_id):
        try:
            return jsonify(jobs.submit(case_id)), 202
        except ValueError as error:
            return jsonify(error=str(error)), 409

    @app.delete('/api/cases/<case_id>')
    def delete_case(case_id):
        with jobs.lock:
            store.get(case_id)
            if case_id in jobs.active:
                return jsonify(error='病例正在分析，请完成后再删除。'), 409
            with store.lock:
                shutil.rmtree(store.directory(case_id))
        return jsonify(deleted=True)

    @app.get('/api/cases/<case_id>/images/<image_id>/<asset>')
    def image_asset(case_id, image_id, asset):
        case = store.get(case_id)
        if not IMAGE_ID.fullmatch(image_id) or not any(x['id'] == image_id for x in case['images']):
            abort(404)
        if asset not in {'preview.png', 'attention.png', 'lime.png', 'occlusion.png', 'fusion.png', 'disagreement.png', 'maps.npz'}:
            abort(404)
        path = store.directory(case_id) / image_id / asset
        if not path.is_file():
            abort(404)
        return send_file(path, as_attachment=asset.endswith('.npz'), download_name=f'{image_id}-{asset}')

    @app.get('/api/cases/<case_id>/download/<kind>')
    def download(case_id, kind):
        case = store.get(case_id)
        if kind == 'report' and case['report']['status'] != 'complete':
            abort(404, description='报告尚未完成。')
        name = {'report': 'case_report.md', 'json': 'case.json'}.get(kind)
        if name is None:
            abort(404)
        path = store.directory(case_id) / name
        if not path.is_file():
            abort(404, description='报告尚未生成。')
        return send_file(path, as_attachment=True, download_name=f'{case_id[:8]}-{name}')

    return app
