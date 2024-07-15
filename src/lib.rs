pub mod config;
pub mod pyscripts;

use std::{error::Error, fmt::Debug, i32};
use pyo3::{prelude::*, types::PyModule};
use config::*;
use pyscripts::*;

#[derive(Clone, Debug)]
pub struct WhisperModel {
    module: Py<PyModule>,
    model: Py<pyo3::PyAny>,
    pub config: WhisperConfig,
}

#[derive(Clone, Debug)]
pub struct Segment{
    pub id: i32,
    pub seek: i32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32,
}

#[derive(Clone)]
pub struct Segments(String, pub Vec<Segment>);

impl ToString for Segments{
    fn to_string(&self) -> String {
        self.0.clone()
    }
}

impl Debug for Segments{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Default for WhisperModel{
    fn default() -> Self {
        return Self::new("base.en".to_string(), "cpu".to_string(), "int8".to_string(), WhisperConfig::default()).unwrap();
    }
}

impl WhisperModel{
    pub fn new(model : String, device : String, compute_type : String, config : WhisperConfig) -> Result<Self, Box<dyn Error>>{
        let m = Python::with_gil(|py|{
            let activators = PyModule::from_code_bound(py, &get_script(), "whisper.py", "Whisper"
            ).unwrap();
            let args = (model, device, compute_type);
            let model = activators.getattr("new_model").unwrap().call1(args).unwrap().unbind();
            return Self{
                module: activators.unbind(),
                model,
                config
            };
        });
        
        return Ok(m);
    }


    fn convert<T : ToString>(value : Option<T>) -> String{
        match value{
            Some(x) => x.to_string(),
            None => "None".to_string(),
        }
    }

    pub fn transcribe(&self, path : String) -> Result<Segments, Box<dyn Error>>{
        let segments = Python::with_gil(|py|{
            let vad = (
                self.config.vad.active,
                self.config.vad.threshold,
                self.config.vad.min_speech_duration,
                Self::convert(self.config.vad.max_speech_duration),
                self.config.vad.min_silence_duration,
                self.config.vad.padding_duration,
            );

            let args = (
                self.model.clone(), 
                path, 
                Self::convert(self.config.starting_prompt.clone()), 
                Self::convert(self.config.prefix.clone()), 
                Self::convert(self.config.language.clone()),
                self.config.beam_size,
                self.config.best_of,
                self.config.patience,
                self.config.length_penalty,
                Self::convert(self.config.chunk_length.clone()),
                vad,
            );

            let pysegments = self.module.getattr(py, "transcribe_audio")
                .unwrap().call1(py, args)?
                .extract::<Vec<(i32, i32, f32, f32, String, f32, f32, f32, f32)>>(py)?;
            let mut segments = Vec::new();
            
            for segment in pysegments{
                segments.push(
                    Segment{
                        id : segment.0,
                        seek : segment.1,
                        start : segment.2,
                        end : segment.3,
                        text : segment.4,
                        temperature : segment.5,
                        avg_logprob : segment.6,
                        compression_ratio : segment.7,
                        no_speech_prob : segment.8,
                    }
                );

            }
            
            return Ok::<Vec<Segment>, Box<dyn Error>>(segments);
        })?;

        let mut text = String::new();
        
        for segment in &segments{
            text.push_str(&segment.text);
        }
        
        return Ok(Segments(text, segments));
    }
}

#[test]
fn create_test(){
    let fw = WhisperModel::default();
    let trans =fw.transcribe(get_path("./man.mp3".to_string())).unwrap();
    assert!(!trans.0.is_empty());
}

pub fn get_path(path : String) -> String{
    let mut new_path = env!("CARGO_MANIFEST_DIR").to_string();
    new_path.push_str(&format!("/src/{}", path));
    return new_path;
}
