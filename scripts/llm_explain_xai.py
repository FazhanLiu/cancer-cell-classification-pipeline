import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cancer_pipeline.artifacts import file_hash, new_run, read_json, save_json
from cancer_pipeline.config import load_environment
from cancer_pipeline.reports import build_prompt, generate_reports, mock_explanation


def main():
    parser = argparse.ArgumentParser(description='Generate resumable reports from XAI evidence; no labels are sent to the LLM.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='outputs/reports')
    parser.add_argument('--provider', choices=['mock', 'deepseek', 'openrouter', 'groq', 'gemini', 'huggingface'], default='mock')
    parser.add_argument('--model', help='Required for real API calls.')
    parser.add_argument('--env-file', default=str(Path(__file__).resolve().parents[1] / '.env'),
                        help='Environment file; defaults to the project root .env. Existing environment values take priority.')
    parser.add_argument('--max-samples', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--sleep', type=float, default=0.5)
    parser.add_argument('--resume-from', help='Previous report directory; reuse successful matching evidence, retry failures.')
    parser.add_argument('--include-review', action='store_true', help='Append true labels/groups locally, never in LLM input.')
    args = parser.parse_args()
    if args.max_samples < 1:
        raise ValueError('max-samples must be positive.')
    if args.provider != 'mock' and not args.model:
        parser.error('--model is required for API providers')
    load_environment(args.env_file)
    if args.provider == 'deepseek' and not os.environ.get('DEEPSEEK_API_KEY', '').strip():
        parser.error('Set DEEPSEEK_API_KEY in your environment or .env before using --provider deepseek')
    samples = read_json(args.input)[:args.max_samples]
    if not samples:
        raise ValueError('No explanation samples.')
    previous = read_json(Path(args.resume_from) / 'llm_explanations.json') if args.resume_from else None
    output = new_run({'output_dir': args.output, 'run_name': 'llm_report', 'parameters': vars(args)}, 'report')
    save_json({'input_hash': file_hash(args.input), 'parameters': vars(args)}, output / 'source.json')
    results = generate_reports(samples, output, args.provider, args.model, args.timeout, args.retries,
                               args.sleep, previous, args.include_review)
    print(f"{output} (failed samples: {sum(x['status'] == 'failed' for x in results)})")

if __name__ == '__main__':
    main()
