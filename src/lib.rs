use std::error::Error;

use pyo3::{prelude::*, types::PyModule};

#[derive(Clone, Debug)]
pub struct WhisperModel {
    module: Py<PyModule>,
    model: Py<pyo3::PyAny>,
}

impl Default for WhisperModel{
    fn default() -> Self {
        return Self::new("base.en".to_string(), "cpu".to_string(), "int8".to_string()).unwrap();
    }
}

impl WhisperModel{
    pub fn new(model : String, device : String, compute_type : String) -> Result<Self, Box<dyn Error>>{
        let m = Python::with_gil(|py|{
            let activators = PyModule::from_code_bound(py, r#"

from faster_whisper import WhisperModel
import os

def new_model(size, device, compute):
    return WhisperModel(size, device=device, compute_type=compute)

def transcribe_audio(model, path, vad=True):
    segments, _ = model.transcribe(audio=path, beam_size=5, vad_filter=vad) #
    segments = list(segments)
    transcript = ""
    
    for segment in segments:
        transcript += segment.text

    print(f"Path: {path}, Transcript: {transcript}")
    os.remove(path)
    return transcript
            "#, "whisper.py", "Whisper"
            ).unwrap();
            let args = (model, device, compute_type);
            let model = activators.getattr("new_model").unwrap().call1(args).unwrap().unbind();
            return Self{
                module: activators.unbind(),
                model,
            };
        });
        
        return Ok(m);
    }

    pub fn transcribe(&self, vad : bool, path : String) -> Result<String, Box<dyn Error>>{
        let transcript = Python::with_gil(|py|{
            let args = (self.model.clone(), path, vad);
            let transcript = self.module.getattr(py, "transcribe_audio").unwrap().call1(py, args);
            
            return transcript;
        })?;

        return Ok(transcript.to_string());
    }
}

#[test]
fn create_test(){
    let fw = WhisperModel::default();
}
