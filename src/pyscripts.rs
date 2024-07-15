
pub fn get_script() -> String{
    r#"

from faster_whisper import WhisperModel
import os

def new_model(size, device, compute):
    return WhisperModel(size, device=device, compute_type=compute)

def transcribe_audio(
    model, 
    path, 
    prompt, 
    prefix, 
    language, 
    beam_size, 
    best_of, 
    patience, 
    length_penalty, 
    chunk_length , 
    vad
):
    vad_par = dict(
        threshold=vad[1],
        min_speech_duration_ms=vad[2],
        max_speech_duration_s= float("inf") if vad[3] == "None" else vad[3] / 100,
        min_silence_duration_ms=vad[4],
        speech_pad_ms=vad[5],
    )

    segments, _ = model.transcribe(
        audio=path, 
        beam_size= beam_size if beam_size > 0 else 5, 
        best_of= best_of if best_of > 0 else 5,
        patience= patience if patience > 0 else 1,
        length_penalty= length_penalty if length_penalty > 0 else 1,
        language= "en" if language == None else language,
        prefix= None if prefix == None else prefix,
        chunk_length= None if chunk_length == "None" else chunk_length,
        initial_prompt= None if prompt == None else prompt,
        vad_filter=vad[0],
        vad_parameters=vad_par,
    )

    segments = list(segments)
    s = []
    
    for segment in segments:
        s.append((
            segment.id,
            segment.seek,
            segment.start,
            segment.end,
            segment.text,
            segment.temperature,
            segment.avg_logprob,
            segment.compression_ratio,
            segment.no_speech_prob,
        ))
    
    return s
        "#.to_string()
}
