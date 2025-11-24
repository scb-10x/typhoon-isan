#!/usr/bin/env python3
"""
Typhoon Whisper (Hugging Face) Inference Script

This script uses the Transformers pipeline for efficient inference.
It supports automatic chunking for long audio files and native timestamp generation.

Usage:
    python inference_whisper.py input_audio.mp3 --device cuda
"""

import argparse
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Define the default model ID
# Replace this with your actual HF repo ID
DEFAULT_MODEL = "scb10x/typhoon-isan-asr-whisper" 

def load_whisper_pipeline(model_id, device='auto'):
    """
    Load the Whisper model and processor into a transformers pipeline.
    """
    if device == 'auto':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"🌪️ Loading Whisper model: {model_id}")
    print(f"   Device: {device} | Precision: {torch_dtype}")

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        # Create the pipeline
        # 'chunk_length_s' enables processing files longer than 30 seconds
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
        return pipe
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Typhoon Whisper Inference")
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace Model ID")
    parser.add_argument("--device", choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument("--language", default="th", help="Language code (default: th)")
    args = parser.parse_args()

    # 1. Load Pipeline
    pipe = load_whisper_pipeline(args.model, args.device)
    if pipe is None:
        return

    print(f"🎙️ Transcribing {args.input_file}...")
    
    # 2. Run Inference
    start_time = time.time()
    
    # The pipeline handles loading, resampling, and chunking internally
    # generate_kwargs forces the model to use specific language
    result = pipe(
        args.input_file, 
        generate_kwargs={"language": args.language, "task": "transcribe"}
    )
    
    end_time = time.time()
    processing_time = end_time - start_time

    # 3. Display Results
    print("\n" + "="*50)
    print(f"📝 Results (Processing Time: {processing_time:.2f}s)")
    print("="*50)
    
    # Print full text
    print(f"Full Text:\n{result['text']}\n")


if __name__ == "__main__":
    main()