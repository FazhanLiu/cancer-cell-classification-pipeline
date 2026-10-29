'use strict';
const $ = id => document.getElementById(id);
const state = {status: null, current: null, files: [], selected: null, map: 'preview', uploading: false, timer: null, polling: false, viewVersion: 0};
const labels = {uploaded:'待分析', queued:'排队中', running:'分析中', complete:'已完成', partial:'部分完成', failed:'未完成', interrupted:'已中断', classifying:'分类中', explaining:'解释中'};
const mapNotes = {
  preview:'原始病理小图预览；模型使用其训练配置中保存的尺寸与归一化进行推理。',
  attention:'Attention Rollout：展示 CLS token 汇聚的区域信息流，本身不区分癌症类别。',
  lime:'Grid LIME：随机遮挡网格并拟合局部代理模型，此图保留支持癌症评分的正向证据。',
  occlusion:'遮挡敏感性：遮住某区域后癌症评分下降越多，该区域的正向证据越突出。',
  fusion:'探索性融合：Attention 50% + LIME 30% + 遮挡 20%，显示归一化后的相对重要性。',
  disagreement:'方法分歧：三种归一化热力图在各位置的标准差；颜色为相对展示，不是校准后的不确定性。'
};
const regions = {'upper-left':'左上部','upper-middle':'上中部','upper-right':'右上部','central-left':'左中部','central-middle':'中央','central-right':'右中部','lower-left':'左下部','lower-middle':'下中部','lower-right':'右下部'};
let toastTimer;
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function toast(message) {
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 6500);
}
async function api(path, options={}) {
  const headers = new Headers(options.headers || {});
  if (options.method && options.method !== 'GET') headers.set('X-Local-Token', document.querySelector('meta[name="local-token"]').content);
  const response = await fetch(path, {...options, headers});
  let data;
  try { data = await response.json(); } catch { throw new Error('本机服务响应异常，请确认服务仍在运行。'); }
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
function showSettings() { if (!$('settings-dialog').open) $('settings-dialog').showModal(); }
function renderStatus() {
  const s = state.status;
  if (!s) return;
  $('readiness').classList.toggle('ready', s.model_ready && s.llm_ready);
  $('readiness-title').textContent = !s.model_ready ? '分类模型尚未就绪，可以先保存病例图像。' : !s.llm_ready ? '分类与区域解释已就绪，报告 API 尚未配置。' : '模型与报告服务已就绪';
  $('readiness-description').textContent = s.model_error || (!s.model_ready ? '配置训练好的 ViT 权重后重启服务，即可启用真实分析。不会使用随机模型生成结果。' : !s.llm_ready ? '图像可先分析；管理员连接报告 API 后，可继续生成报告。' : `${s.model.name} · 阈值 ${s.model.threshold.toFixed(3)} · ${s.model.device.toUpperCase()} · ${s.llm_model}`);
  $('settings-state').replaceChildren(element('span', `pill ${s.model_ready?'complete':'partial'}`, s.model_ready?'ViT 模型已加载':'模型未加载'), element('span', `pill ${s.llm_ready?'complete':'partial'}`, s.llm_ready?'报告 API 已配置':'报告 API 未配置'));
  renderPending();
}
async function loadHistory() {
  const data = await api('/api/cases');
  $('history-count').textContent = data.cases.length;
  $('history-list').replaceChildren();
  if (!data.cases.length) { $('history-list').append(element('p','history-empty','还没有病例\n上传第一组病理图像开始。')); return; }
  for (const c of data.cases) {
    const button = element('button',`history-item ${state.current?.id===c.id?'selected':''}`);
    button.append(element('strong','',c.reference), element('small','',`${c.image_count} 张图像 · ${labels[c.status] || c.status}`));
    button.addEventListener('click', () => openCase(c.id));
    $('history-list').append(button);
  }
}
function clearPending() { for (const item of state.files) if (item.url) URL.revokeObjectURL(item.url); state.files = []; renderPending(); }
function renderPending() {
  $('pending-files').replaceChildren();
  for (const item of state.files) {
    const card = element('div','pending-item');
    if (item.url) { const img = element('img'); img.src=item.url; img.alt='待上传图像预览'; card.append(img); }
    else card.append(element('span','file-placeholder','TIFF'));
    const description = element('div');
    description.append(element('strong','',item.file.name),element('small','',`${(item.file.size/1024/1024).toFixed(2)} MB`));
    const remove = element('button','', '×'); remove.setAttribute('aria-label',`移除 ${item.file.name}`);
    remove.addEventListener('click', () => { if (item.url) URL.revokeObjectURL(item.url); state.files=state.files.filter(x=>x!==item); renderPending(); });
    card.append(description,remove); $('pending-files').append(card);
  }
  $('selected-count').textContent = state.files.length ? `已选择 ${state.files.length} 张图像 · 同一个病例` : '尚未选择图片';
  $('metric-total').replaceChildren(document.createTextNode(String(state.current ? state.current.images.length : state.files.length)),element('small','','张'));
  const disabled = !state.files.length || state.uploading;
  $('clear-files').disabled=disabled; $('save-draft').disabled=disabled;
  $('upload-analyze').disabled=disabled || !state.status?.model_ready;
  $('upload-analyze').title=state.status?.model_ready?'':'请先配置训练好的 ViT 权重';
  $('new-case').disabled=state.uploading;
}
function addFiles(files) {
  if (state.uploading) return;
  let skipped=0;
  for (const file of files) {
    if (!/\.(png|jpe?g|tiff?|webp)$/i.test(file.name) || file.size>10*1024*1024 || !file.size || state.files.length>=20) { skipped++; continue; }
    if (state.files.some(x=>x.file.name===file.name && x.file.size===file.size && x.file.lastModified===file.lastModified)) continue;
    state.files.push({file, url:/\.tiff?$/i.test(file.name)?null:URL.createObjectURL(file)});
  }
  if (skipped) toast(`${skipped} 个文件未加入：请使用支持的图片格式，每张不超过 10 MB，每批最多 20 张。`);
  renderPending(); $('file-input').value='';
}
function newCase() {
  if (state.uploading) return;
  state.viewVersion++; clearInterval(state.timer); state.timer=null; state.current=null; state.selected=null; state.map='preview';
  localStorage.removeItem('pathscope-case'); clearPending();
  $('case-reference').value=''; $('upload-panel').hidden=false; $('case-panel').hidden=true;
  $('page-title').textContent='让每一张图像，都有可追溯的解释。'; $('page-subtitle').textContent='一个病例，一组病理小图。从模型分类到区域解释，再到中文分析报告。';
  for (const id of ['metric-positive','metric-negative','metric-progress']) $(id).textContent='—';
  $('metric-stage').textContent='等待上传'; updateWorkflow(); loadHistory().catch(e=>toast(e.message));
}
async function upload(start) {
  if (state.uploading || !state.files.length) return;
  state.uploading=true; renderPending();
  const form = new FormData(); form.append('reference',$('case-reference').value.trim());
  for (const {file} of state.files) form.append('images',file);
  try {
    const c = await api('/api/cases',{method:'POST',body:form});
    state.uploading=false; clearPending(); await openCase(c.id);
    if (start) await analyze(); else toast('病例已保存在本机，模型配置完成后可开始分析。');
  } catch(e) { toast(e.message); }
  finally { state.uploading=false; renderPending(); }
}
async function openCase(id) {
  if (state.uploading) return;
  try {
    const version=++state.viewVersion; clearInterval(state.timer); state.timer=null;
    const c=await api(`/api/cases/${id}`);
    if(version!==state.viewVersion)return;
    state.current=c; state.selected=c.images[0]?.id; state.map='preview';
    localStorage.setItem('pathscope-case',id);
    $('upload-panel').hidden=true; $('case-panel').hidden=false;
    renderCase(); await loadHistory(); beginPolling();
  } catch(e) { toast(e.message); }
}
function beginPolling() {
  clearInterval(state.timer); state.timer=null;
  if (!['queued','running'].includes(state.current?.status)) return;
  state.timer=setInterval(async()=>{
    if (state.polling || !state.current) return;
    const id=state.current.id; state.polling=true;
    try {
      const c=await api(`/api/cases/${id}`);
      if (state.current?.id!==id) return;
      const changed=state.current.status!==c.status; state.current=c; renderCase();
      if (changed) await loadHistory();
      if (!['queued','running'].includes(c.status)) { clearInterval(state.timer); state.timer=null; }
    } catch(e) { clearInterval(state.timer); state.timer=null; toast(e.message); }
    finally { state.polling=false; }
  },1500);
}
async function analyze() {
  const id=state.current?.id; if (!id) return;
  $('analyze-case').disabled=true;
  try { const c=await api(`/api/cases/${id}/analyze`,{method:'POST'}); if(state.current?.id!==id)return; state.current=c; renderCase(); await loadHistory(); beginPolling(); }
  catch(e) { toast(e.message); renderCase(); }
}
function updateWorkflow() {
  const c=state.current;
  const allPred=c && c.images.every(x=>x.prediction);
  const allXai=c && c.images.every(x=>x.status==='complete');
  const values={upload:c?'done':'current', classify:allPred?'done':c?'current':'', xai:allXai?'done':c?.images.some(x=>x.prediction)?'current':'', report:c?.report.status==='complete'?'done':c?.report.status==='generating'?'current':''};
  for (const [name,value] of Object.entries(values)) $(`step-${name}`).className=`workflow-step ${value}`;
}
function renderCase() {
  const c=state.current; if(!c)return;
  $('page-title').textContent=c.reference;
  $('page-subtitle').textContent=`病例 ${c.id.slice(0,8).toUpperCase()} · ${c.images.length} 张组织病理小图 · 逐图分析与病例汇总`;
  const predictions=c.images.filter(x=>x.prediction); const positive=predictions.filter(x=>x.prediction.pred_label===1).length;
  const done=c.images.filter(x=>x.status==='complete').length;
  $('metric-total').replaceChildren(document.createTextNode(String(c.images.length)),element('small','','张'));
  $('metric-positive').textContent=predictions.length?positive:'—'; $('metric-negative').textContent=predictions.length?predictions.length-positive:'—';
  $('metric-progress').textContent=`${done}/${c.images.length}`;
  $('metric-stage').textContent=c.report.status==='generating'?'图像已处理 · 报告生成中':c.status==='complete'?'图像与报告均完成':`图像解释完成 · ${labels[c.status] || c.status}`;
  $('case-status').className=`pill ${c.status}`; $('case-status').textContent=labels[c.status]||c.status;
  $('case-created').textContent=new Date(c.created_utc).toLocaleString('zh-CN',{hour12:false});
  $('job-message').textContent=c.message; $('job-message').classList.toggle('busy-pulse',['queued','running'].includes(c.status));
  const busy=['queued','running'].includes(c.status);
  $('analyze-case').disabled=busy || c.status==='complete' || !state.status?.model_ready;
  $('analyze-case').textContent=busy?'正在分析…':c.status==='uploaded'?'开始分析 →':c.status==='complete'?'分析已完成':'继续分析 / 重试 →';
  $('delete-case').disabled=busy;
  $('model-label').textContent=c.model?.name||'ViT · 待加载';
  $('download-json').href=`/api/cases/${c.id}/download/json`;
  $('image-strip').replaceChildren();
  c.images.forEach((image,index)=>{
    const button=element('button',`image-item ${image.id===state.selected?'active':''}`);
    const img=element('img'); img.src=`/api/cases/${c.id}/images/${image.id}/preview.png`; img.alt=`图像 ${index+1}`;
    button.append(img,element('small','',`图像 ${String(index+1).padStart(2,'0')}`),element('span',`thumb-status ${image.status}`));
    button.addEventListener('click',()=>{state.selected=image.id; renderCase();}); $('image-strip').append(button);
  });
  $('results-table').replaceChildren();
  for(const image of c.images){
    const row=element('tr'); const name=element('td','',image.id); name.append(element('small','',image.filename));
    const label=element('td'); label.append(element('span',`pill ${image.prediction?(image.prediction.pred_label?'positive':'negative'):''}`,image.prediction?(image.prediction.pred_label?'模型阳性':'模型阴性'):'等待分类'));
    row.append(name,label,element('td','',image.prediction?image.prediction.prob_cancer.toFixed(4):'—'),element('td','',image.prediction?image.prediction.threshold.toFixed(3):'—'),element('td','',labels[image.status]||image.status)); $('results-table').append(row);
  }
  renderImage(); renderReport(); updateWorkflow();
}
function renderImage() {
  const c=state.current; const image=c?.images.find(x=>x.id===state.selected); if(!image)return;
  if(image.status!=='complete')state.map='preview';
  for(const button of $('map-tabs').querySelectorAll('button')){button.disabled=button.dataset.map!=='preview' && image.status!=='complete'; button.classList.toggle('active',button.dataset.map===state.map); button.setAttribute('aria-selected',String(button.dataset.map===state.map));}
  const url=`/api/cases/${c.id}/images/${image.id}/${state.map}.png`;
  if($('main-image').getAttribute('src')!==url)$('main-image').src=url;
  $('main-image').alt=`${image.id} · ${state.map==='preview'?'原图':state.map+' 区域解释'}`;
  $('viewer-caption').textContent=image.id; $('map-description').textContent=mapNotes[state.map];
  $('selected-name').textContent=image.filename; $('selected-dimensions').textContent=`${image.width} × ${image.height} px · ${image.id}`;
  const p=image.prediction; $('prediction-detail').hidden=!p;
  $('prediction-badge').className=`pill ${p?(p.pred_label?'positive':'negative'):''}`;
  $('prediction-badge').textContent=p?(p.pred_label?'模型阳性':'模型阴性'):'等待分类';
  if(p){$('selected-score').textContent=p.prob_cancer.toFixed(4); $('score-bar').value=p.prob_cancer; $('selected-threshold').textContent=`分类阈值 ${p.threshold.toFixed(3)} · 距阈值 ${p.score_margin.toFixed(4)} · 非校准置信度`;}
  $('selected-error').hidden=!image.error; $('selected-error').textContent=image.error||'';
  $('regions-content').replaceChildren();
  if(!image.evidence){$('regions-content').append(element('div','empty-evidence','三种方法解释完成后，将显示各自的重点区域和融合摘要。'));return;}
  for(const [title,key] of [['融合重点区域','top_final_score_regions'],['Attention 信息流','top_attention_regions'],['LIME 正向证据','top_lime_regions'],['遮挡正向证据','top_occlusion_regions']]){
    const section=element('section','region-section'); section.append(element('h3','',title)); const rows=image.evidence[key]||[];
    if(!rows.length)section.append(element('div','region-empty','未得到非零的正向区域摘要。'));
    for(const r of rows){const row=element('div','region-row');row.append(element('span','',regions[r.region]||r.region),element('strong','',r.score.toFixed(3)));section.append(row);}
    $('regions-content').append(section);
  }
  const difference=element('div','region-section');difference.append(element('h3','','平均方法分歧'),element('div','region-row',`${image.evidence.mean_method_disagreement.toFixed(4)} · 归一化图间差异`));$('regions-content').append(difference);
}
function renderReport() {
  const c=state.current; const r=c.report;
  $('report-status').className=`pill ${r.status==='complete'?'complete':r.status==='failed'?'partial':r.status==='generating'?'running':''}`;
  $('report-status').textContent={not_started:'等待分析',generating:'正在生成',complete:'报告已生成',failed:'待重试'}[r.status]||r.status;
  $('download-report').hidden=r.status!=='complete'; $('download-report').href=`/api/cases/${c.id}/download/report`;
  $('report-content').replaceChildren(); $('report-content').className='report-content';
  if(r.status==='complete'){$('report-content').textContent=r.text;return;}
  $('report-content').classList.add('empty-report');
  const title=r.status==='failed'?'报告暂未生成':r.status==='generating'?'正在将图像证据整理为中文报告':'让结果形成一份清晰的病例摘要';
  const note=r.status==='failed'?(r.error||'可点击继续分析重试。'):r.status==='generating'?'报告模型仅接收去标识化的预测与区域摘要，请稍候。':'图像解释完成后自动生成中文病例报告，包含逐图结果、方法分歧与复核提示。';
  $('report-content').append(element('span','report-glyph','≡'),element('h3','',title),element('p','',note));
}
$('dropzone').addEventListener('click',()=>$('file-input').click());
$('dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('file-input').click();}});
for(const event of ['dragover','dragenter'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.add('dragging');});
for(const event of ['dragleave','drop'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.remove('dragging');});
$('dropzone').addEventListener('drop',e=>addFiles(e.dataTransfer.files));
$('file-input').addEventListener('change',e=>addFiles(e.target.files));
$('clear-files').addEventListener('click',clearPending);
$('new-case').addEventListener('click',newCase);
$('save-draft').addEventListener('click',()=>upload(false)); $('upload-analyze').addEventListener('click',()=>upload(true));
$('analyze-case').addEventListener('click',analyze);
$('delete-case').addEventListener('click',async()=>{const c=state.current;if(!c||!confirm(`删除病例「${c.reference}」及本机保存的图片和结果？`))return;try{await api(`/api/cases/${c.id}`,{method:'DELETE'});newCase();toast('病例及相关文件已删除。');}catch(e){toast(e.message);}});
for(const id of ['open-settings','readiness-settings'])$(id).addEventListener('click',showSettings);
for(const id of ['close-settings','close-settings-bottom'])$(id).addEventListener('click',()=>$('settings-dialog').close());
$('settings-dialog').addEventListener('click',e=>{if(e.target===$('settings-dialog')){const b=$('settings-dialog').getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)$('settings-dialog').close();}});
$('map-tabs').addEventListener('click',e=>{const button=e.target.closest('button[data-map]');if(button&&!button.disabled){state.map=button.dataset.map;renderImage();}});
window.addEventListener('beforeunload',()=>{for(const item of state.files)if(item.url)URL.revokeObjectURL(item.url);});
(async()=>{try{state.status=await api('/api/status');renderStatus();await loadHistory();const id=localStorage.getItem('pathscope-case');if(id)await openCase(id);}catch(e){$('readiness-title').textContent='无法连接本机服务';$('readiness-description').textContent=e.message;toast(e.message);}})();
