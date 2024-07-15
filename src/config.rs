use derive_builder::Builder;

#[derive(Builder, Clone, Debug, Default)]
#[builder(setter(into))]
pub struct WhisperConfig{
    #[builder(default = "None")]
    pub starting_prompt : Option<String>,
    #[builder(default = "None")]
    pub prefix : Option<String>,
    #[builder(default = "VadConfig::default()")]
    pub vad : VadConfig,
    #[builder(default = "None")]
    pub language : Option<String>,
    #[builder(default = "5")]
    pub beam_size : usize,
    #[builder(default = "5")]
    pub best_of : usize,
    #[builder(default = "1.0")]
    pub patience : f32,
    #[builder(default = "1.0")]
    pub length_penalty : f32,
    #[builder(default = "None")]
    pub chunk_length : Option<usize>,
}

#[derive(Builder, Clone, Debug, Default)]
#[builder(setter(into))]
pub struct VadConfig{
    #[builder(default = "false")]
    pub active : bool,
    #[builder(default = "0.5")]
    pub threshold : f32,
    #[builder(default = "250")]
    pub min_speech_duration : i32,
    #[builder(default = "None")]
    pub max_speech_duration : Option<i32>,
    #[builder(default = "2000")]
    pub min_silence_duration : i32,
    #[builder(default = "400")]
    pub padding_duration : i32,
}
