pub mod audio;
pub mod config;
pub mod models;

use config::*;
use ct2rs::{Config, Whisper, WhisperOptions};
use std::{error::Error, fmt::Debug, i32, path::PathBuf};
use symphonia::core::sample;

use crate::{
    audio::parse_audio_file,
    models::{ModelHandler, ModelHandlerBuilder},
};

#[derive(Debug)]
pub struct WhisperModel {
    pub model: Whisper,
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

#[derive(Clone)]
pub struct Segments {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<Segment>,
}

impl ToString for Segments {
    fn to_string(&self) -> String {
        self.text.clone()
    }
}

impl Debug for Segments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

pub fn get_path(path: String) -> String {
    let mut new_path = env!("CARGO_MANIFEST_DIR").to_string();
    new_path.push_str(&format!("/src/{}", path));
    return new_path;
}

impl WhisperModel {
    pub async fn new(config: Option<Config>) -> Result<Self, Box<dyn Error>> {
        Self::new_with_handler(ModelHandlerBuilder::default().build().await, config)
    }

    pub fn new_with_handler(
        handler: ModelHandler,
        config: Option<Config>,
    ) -> Result<Self, Box<dyn Error>> {
        let whisper = Whisper::new(
            handler.get_model_dir(),
            match config {
                Some(x) => x,
                None => Config::default(),
            },
        )?;

        Ok(Self { model: whisper })
    }

    pub fn generate(
        &self,
        samples: &[f32],
        options: Option<WhisperOptions>,
    ) -> Result<Segments, Box<dyn Error>> {
        for sample in samples.iter().filter(|x| **x < -1.0 || **x > 1.0) {
            println!("{:?}", sample);
        }

        let res = self.model.generate(
            samples,
            None,
            true,
            &match options {
                Some(x) => x,
                None => Default::default(),
            },
        )?;

        let mut segments = Vec::new();
        let mut text = String::new();

        for r in res {
            let mut start = None;
            let mut txt: Option<&str> = None;
            for r in r.split(|x| x == '<' || x == '>') {
                // println!("{r:?}");
                match r.trim().chars().nth(0) {
                    Some(x) if x == '|' => {
                        let num = r
                            .trim()
                            .strip_prefix('|')
                            .unwrap()
                            .strip_suffix('|')
                            .unwrap()
                            .trim()
                            .parse()?;
                        if let None = start {
                            start = Some(num);
                        } else {
                            segments.push(Segment {
                                start: start.unwrap(),
                                end: num,
                                text: txt.unwrap().to_string(),
                            });
                            text.push_str(&format!(" {}", txt.unwrap()));
                            start = None;
                            txt = None;
                        }
                    }
                    Some(_) => {
                        if let Some(t) = txt {
                            segments.push(Segment {
                                start: start.unwrap_or(0.0),
                                end: 0.0,
                                text: t.to_string(),
                            });
                            text.push_str(&format!(" {}", t));
                            start = None;
                            txt = None;
                        } else {
                            txt = Some(r)
                        }
                    }
                    None => {}
                };
            }
        }

        Ok(Segments {
            text: text.trim().to_string(),
            segments,
            language: None,
        })
    }
}
