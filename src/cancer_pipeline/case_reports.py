"""One real DeepSeek report per case; no filenames, patient names or images sent."""
import json
from datetime import datetime, timezone
from pathlib import Path

from cancer_pipeline.artifacts import object_hash, read_json, save_json
from cancer_pipeline.reports import DEFAULT_SYSTEM_PROMPT, _safe_error, call_llm, compatible_endpoint, evidence_for


def case_evidence(case):
    images = []
    for item in case['images']:
        if item.get('prediction'):
            evidence = item.get('evidence') or {'id': item['id'], **item['prediction']}
            images.append({**evidence_for(evidence), 'xai_status': item['status']})
    positive = sum(x['pred_label'] for x in images)
    return {'uploaded_images': len(case['images']), 'classified_images': len(images),
            'model_positive_images': positive, 'model_negative_images': len(images) - positive,
            'explanation_complete_images': sum(x['status'] == 'complete' for x in case['images']),
            'failed_image_ids': [x['id'] for x in case['images'] if x['status'] != 'complete'],
            'images': images,
            'limits': ['同一病例的多个病理 patch，非完整切片或病灶分割。',
                       '阳性图像数量不能换算为病例患癌概率；模型分数未必经过校准。',
                       'XAI 展示模型证据，不证明癌细胞位置。Attention 不区分类别。']}


def generate_case_report(case, output_dir, model, timeout=120, caller=None, provider='deepseek'):
    evidence = case_evidence(case)
    if not evidence['classified_images']:
        raise ValueError('没有成功分类的图片，无法生成病例报告。')
    prompt = ('请用中文整理下面的病理图像辅助分析证据，使用纯文本小标题与分段。'
              '依次输出：病例图像概览、逐图模型结果、区域解释与方法分歧、需要复核的图像、分析局限。'
              '逐图用 image_001 等编号，不要编造患者信息。不得推断细胞形态、具体病灶、分期或治疗。'
              '不得将热力图解释为病灶分割，不得将图片阳性比例或最高分当作病例患癌概率。'
              '阴性图片的 LIME/遮挡图仍解释癌症评分正向证据，不是证明正常的区域。'
              '有失败或缺失解释时明确说明。最终结论应由医生结合原始切片复核。\n\n' +
              json.dumps(evidence, ensure_ascii=False, allow_nan=False))
    fingerprint = object_hash({'evidence': evidence, 'prompt': prompt, 'model': model,
                               'system': DEFAULT_SYSTEM_PROMPT, 'provider': provider,
                               **({'endpoint': compatible_endpoint()} if provider == 'compatible' else {})})
    output = Path(output_dir)
    path = output / 'case_report.json'
    if path.exists():
        previous = read_json(path)
        if previous.get('fingerprint') == fingerprint and previous.get('status') == 'complete':
            return previous
    report = {'max_output_tokens': 4096, 'provider': provider, 'model': model, 'fingerprint': fingerprint,
              'evidence': evidence, 'prompt': prompt, 'system_prompt': DEFAULT_SYSTEM_PROMPT,
              'generated_utc': datetime.now(timezone.utc).isoformat()}
    try:
        text = caller(provider, prompt, model, timeout) if caller else call_llm(provider, prompt, model, timeout, max_tokens=4096)
        if not isinstance(text, str) or not text.strip():
            raise ValueError('报告 API 返回了空内容。')
        report.update(status='complete', text=text.strip())
    except Exception as error:
        report.update(status='failed', error=_safe_error(error))
    save_json(report, path)
    if report['status'] == 'complete':
        # Case reference is local only: it is intentionally absent from the prompt.
        title = case['reference'].replace('\n', ' ').replace('\r', ' ')
        header = f"# {title} · 病理图像辅助分析\n\n病例编号：{case['id']}\n\n"
        header += '研究用模型结果，需医生复核；不替代病理诊断。\n\n'
        (output / 'case_report.md').write_text(header + report['text'], encoding='utf-8')
    return report
