pub mod audio;
pub mod config;
pub mod models;

use config::*;
use ct2rs::{Whisper, WhisperOptions};
use pyo3::{prelude::*, types::PyModule};
use std::str::FromStr;
use std::{error::Error, fmt::Debug, i32};

#[derive(Debug)]
pub struct WhisperModel {
    model: Whisper,
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub temperature: f32,
    pub no_speech_prob: f32,
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
