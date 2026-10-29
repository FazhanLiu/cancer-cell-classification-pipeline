"""Report generation from structured evidence, without labels in model input."""
import json
import math
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path

from cancer_pipeline.artifacts import object_hash, read_json, save_json
from cancer_pipeline.metrics import validate_predictions, validate_threshold

DEFAULT_SYSTEM_PROMPT = """You describe research model outputs from supplied structured evidence only.
Do not infer cell morphology, pathology subtype, or lesions from heatmap summaries.
The heatmap is relative model importance, not a segmentation or localization guarantee.
Attention rollout is class-independent information flow; LIME and occlusion target cancer score.
Fusion weights are exploratory. Score margin is not calibrated confidence.
Describe disagreements and missing/zero evidence honestly. Do not present clinical diagnosis.
Return a concise prediction description, evidence summary, limitations, and research-only note."""


def request_json(url, headers, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc


def call_openrouter(prompt, model, timeout):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    response = request_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/FazhanLiu/cancer-cell-classification-pipeline",
            "X-Title": "Cancer Cell Classification Pipeline",
        },
        {
            "model": model or "meta-llama/llama-3.1-8b-instruct:free",
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout,
    )
    return response["choices"][0]["message"]["content"].strip()


def call_groq(prompt, model, timeout):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    response = request_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        {
            "model": model or "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout,
    )
    return response["choices"][0]["message"]["content"].strip()


def call_deepseek(prompt, model, timeout, max_tokens=1024):
    """Use DeepSeek's official non-streaming Chat Completions endpoint."""
    api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('DEEPSEEK_API_KEY is not set.')
    response = request_json(
        'https://api.deepseek.com/chat/completions',
        {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        {'model': model,
         'messages': [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT},
                      {'role': 'user', 'content': prompt}],
         'thinking': {'type': 'disabled'}, 'stream': False,
         'temperature': 0.2, 'max_tokens': max_tokens},
        timeout,
    )
    choices = response.get('choices', [])
    if not choices:
        raise RuntimeError('DeepSeek returned no completion choices.')
    choice = choices[0]
    if choice.get('finish_reason') != 'stop':
        raise RuntimeError(f"DeepSeek completion did not finish normally: {choice.get('finish_reason')}")
    content = choice.get('message', {}).get('content')
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('DeepSeek returned empty report content.')
    return content.strip()


def call_gemini(prompt, model, timeout):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    gemini_model = model or "gemini-1.5-flash"
    response = request_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}",
        {"Content-Type": "application/json"},
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{DEFAULT_SYSTEM_PROMPT}\n\n{prompt}"}],
                }
            ],
            "generationConfig": {"temperature": 0.2},
        },
        timeout,
    )
    return response["candidates"][0]["content"]["parts"][0]["text"].strip()


def call_huggingface(prompt, model, timeout):
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        raise RuntimeError("HF_TOKEN is not set.")
    hf_model = model or "mistralai/Mistral-7B-Instruct-v0.3"
    response = request_json(
        f"https://api-inference.huggingface.co/models/{hf_model}",
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        {
            "inputs": f"[INST] {DEFAULT_SYSTEM_PROMPT}\n\n{prompt} [/INST]",
            "parameters": {
                "max_new_tokens": 350,
                "temperature": 0.2,
                "return_full_text": False,
            },
        },
        timeout,
    )
    if isinstance(response, list) and response:
        return response[0].get("generated_text", "").strip()
    return json.dumps(response, ensure_ascii=False)


def report_settings():
    """Resolve hosted report settings without exposing credentials."""
    provider = os.environ.get('LLM_PROVIDER', '').strip() or 'deepseek'
    if provider not in {'deepseek', 'compatible'}:
        raise ValueError('LLM_PROVIDER must be deepseek or compatible')
    model = os.environ.get('LLM_MODEL', '').strip()
    if provider == 'deepseek':
        model = model or os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash')
        ready = bool(os.environ.get('DEEPSEEK_API_KEY', '').strip())
    else:
        ready = bool(model and os.environ.get('LLM_API_KEY', '').strip()
                     and os.environ.get('LLM_BASE_URL', '').strip())
    return {'provider': provider, 'model': model, 'ready': ready}


def compatible_endpoint():
    base = os.environ.get('LLM_BASE_URL', '').strip().rstrip('/')
    parts = urlsplit(base)
    if (parts.scheme != 'https' or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment):
        raise ValueError('LLM_BASE_URL must be an HTTPS API base URL without credentials or query parameters')
    return base if base.endswith('/chat/completions') else base + '/chat/completions'


def call_compatible(prompt, model, timeout, max_tokens=1024):
    """One configurable Chat Completions adapter for compatible API vendors."""
    key = os.environ.get('LLM_API_KEY', '').strip()
    if not key or not model:
        raise RuntimeError('LLM_API_KEY and LLM_MODEL are required.')
    response = request_json(compatible_endpoint(),
        {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        {'model': model, 'messages': [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT},
                                     {'role': 'user', 'content': prompt}],
         'stream': False, 'max_tokens': max_tokens}, timeout)
    choices = response.get('choices', [])
    if not choices or choices[0].get('finish_reason') != 'stop':
        raise RuntimeError('Report API returned an incomplete completion.')
    content = choices[0].get('message', {}).get('content')
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('Report API returned empty content.')
    return content.strip()


def call_llm(provider, prompt, model, timeout, max_tokens=1024):
    if provider == "mock":
        raise RuntimeError("mock provider should be handled before call_llm.")
    if provider == "openrouter":
        return call_openrouter(prompt, model, timeout)
    if provider == "groq":
        return call_groq(prompt, model, timeout)
    if provider == "deepseek":
        return call_deepseek(prompt, model, timeout, max_tokens=max_tokens)
    if provider == 'compatible':
        return call_compatible(prompt, model, timeout, max_tokens=max_tokens)
    if provider == "gemini":
        return call_gemini(prompt, model, timeout)
    if provider == "huggingface":
        return call_huggingface(prompt, model, timeout)
    raise ValueError(f"Unsupported provider: {provider}")




def confidence_band(prob_cancer, threshold):
    # Compatibility name: this deliberately returns a numeric margin, not confidence.
    return abs(float(prob_cancer) - float(threshold))


def evidence_for(sample):
    validate_threshold(sample['threshold'])
    validate_predictions([sample['pred_label']], [sample['prob_cancer']])
    if sample['pred_label'] != int(sample['prob_cancer'] >= sample['threshold']):
        raise ValueError('Report prediction does not match its decision threshold.')
    fields = ('id', 'model', 'prob_cancer', 'pred_label', 'threshold', 'target',
              'top_attention_regions', 'top_lime_regions', 'top_occlusion_regions',
              'top_final_score_regions', 'fusion_weights', 'fusion_status',
              'mean_method_disagreement', 'relative_high_score_area_ratio_0_7', 'source', 'decision')
    evidence = {k: sample[k] for k in fields if k in sample}
    evidence['score_margin'] = confidence_band(sample['prob_cancer'], sample['threshold'])
    evidence['limitations'] = ['No morphology evidence supplied', 'Not lesion segmentation',
                               'Attention is class-independent', 'Score margin is not calibrated confidence']
    return evidence


def build_prompt(sample):
    return 'Describe only this supplied model evidence:\n' + json.dumps(evidence_for(sample), ensure_ascii=False, allow_nan=False)


def mock_explanation(sample):
    evidence = evidence_for(sample)
    label = 'cancer' if evidence['pred_label'] else 'non-cancer'
    return (f"The model predicts {label}, with cancer score {evidence['prob_cancer']:.4f} "
            f"and threshold {evidence['threshold']:.4f}. Score margin: {evidence['score_margin']:.4f}; "
            'this is not calibrated confidence. '
            f"Class-independent attention regions: {evidence.get('top_attention_regions', [])}. "
            f"Positive cancer-score LIME evidence: {evidence.get('top_lime_regions', [])}. "
            f"Positive cancer-score occlusion evidence: {evidence.get('top_occlusion_regions', [])}. "
            f"Exploratory fusion regions: {evidence.get('top_final_score_regions', [])}. "
            f"Mean method disagreement: {evidence.get('mean_method_disagreement', 'unavailable')}. "
            'These are relative importance summaries, not lesion locations or cell morphology. Research use only.')


def make_markdown_report(results):
    lines = ['# Model evidence report', '', 'Relative importance is not lesion segmentation. Research use only.', '']
    for item in results:
        lines += [f"## {item['id']}", '', f"Status: {item['status']}; model: {item['model']}", '',
                  item.get('explanation') or item.get('error', 'No result'), '']
        if item.get('offline_review'):
            lines += ['Offline review (not supplied to the LLM): ' + json.dumps(item['offline_review']), '']
    return '\n'.join(lines)


def _safe_error(error):
    message = f'{type(error).__name__}: {error}'
    for key in ('LLM_API_KEY', 'OPENROUTER_API_KEY', 'GROQ_API_KEY', 'DEEPSEEK_API_KEY', 'GEMINI_API_KEY', 'HF_TOKEN'):
        if os.environ.get(key):
            message = message.replace(os.environ[key], '[REDACTED]')
            if os.environ[key].strip():
                message = message.replace(os.environ[key].strip(), '[REDACTED]')
    return message


def generate_reports(samples, output, provider='mock', model=None, timeout=60, retries=2,
                     sleep=0.5, previous=None, include_review=False, caller=None):
    if provider != 'mock' and not model:
        raise ValueError('API providers require an explicit --model; model availability is provider-specific.')
    if not 0 <= retries <= 10 or not all(math.isfinite(v) for v in (timeout, sleep)) or timeout <= 0 or sleep < 0:
        raise ValueError('Invalid retry, timeout, or delay settings.')
    model = model or 'template-v2'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Report sample IDs must be unique.')
    existing = {x['id']: x for x in (previous or [])}
    results = []
    for sample in samples:
        evidence = evidence_for(sample)
        prompt = build_prompt(sample)
        fingerprint = object_hash({'evidence': evidence, 'system': DEFAULT_SYSTEM_PROMPT, 'provider': provider, 'model': model,
                                   **({'endpoint': compatible_endpoint()} if provider == 'compatible' else {})})
        old = existing.get(sample['id'])
        if old and old.get('fingerprint') == fingerprint and old.get('status') in ('mock', 'api'):
            item = dict(old)
            item.pop('offline_review', None)
        else:
            item = {'id': sample['id'], 'provider': provider, 'model': model, 'status': 'pending',
                    'fingerprint': fingerprint, 'evidence': evidence, 'system_prompt': DEFAULT_SYSTEM_PROMPT,
                    'prompt': prompt, 'attempts': 0}
            for attempt in range(retries + 1):
                item['attempts'] = attempt + 1
                try:
                    result = mock_explanation(sample) if provider == 'mock' else (caller or call_llm)(provider, prompt, model, timeout)
                    if not isinstance(result, str) or not result.strip():
                        raise ValueError('Empty LLM response.')
                    item.update(status='mock' if provider == 'mock' else 'api', explanation=result)
                    item.pop('error', None)
                    break
                except Exception as error:
                    item.update(status='failed', error=_safe_error(error))
                    save_json(results + [item], output / 'llm_explanations.json')
                    if attempt < retries:
                        time.sleep(sleep)
            if provider != 'mock':
                time.sleep(sleep)
        if include_review:
            item['offline_review'] = {k: sample[k] for k in ('true_label', 'group') if k in sample}
        results.append(item)
        save_json(results, output / 'llm_explanations.json')
        (output / 'llm_explanations.md').write_text(make_markdown_report(results), encoding='utf-8')
    return results
