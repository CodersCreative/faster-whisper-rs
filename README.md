# faster-whisper-rs

#### It is a rust crate for easily implementing Speech-To-Text into your rust programs.

### Python api:
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)

### Install Rust

[Install Rust](https://www.rust-lang.org/tools/install)

On Linux or MacOS:
```
curl --proto '=https' --tlsv1.2 -ssf https://sh.rustup.rs | sh
```
### Example of making a transcript

```Rust
use std::error::Error;
use faster-whisper-rs::WhisperModel;

fn main() -> Result<(), Box<dyn Error>>{
    let fw = WhisperModel::default();
    let transcript = fw.transcribe("Path to file".to_string())?;

    println!("{}", transcript);

    Ok(())
}

```
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
