//! Audio device discovery and selection.

use cpal::traits::{DeviceTrait, HostTrait};

use crate::error::{Error, Result};

/// Information about an available audio input device.
#[derive(Debug, Clone)]
pub struct AudioDevice {
    /// Name of the audio device
    pub name: String,
    /// Whether this is the default input device
    pub is_default: bool,
}

/// Lists all available audio input devices.
///
/// # Example
///
/// ```no_run
/// use bpm_analyzer::list_audio_devices;
///
/// let devices = list_audio_devices()?;
/// for device in devices {
///     println!("{} {}", device.name, if device.is_default { "(default)" } else { "" });
/// }
/// # Ok::<(), bpm_analyzer::Error>(())
/// ```
pub fn list_audio_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let default_device = host.default_input_device();
    let default_name = default_device
        .as_ref()
        .and_then(|d| d.description().ok())
        .map(|desc| desc.name().to_string());

    let devices = host
        .input_devices()?
        .filter_map(|device| {
            device.description().ok().map(|desc| {
                let name = desc.name().to_string();
                let is_default = default_name.as_ref() == Some(&name);
                AudioDevice { name, is_default }
            })
        })
        .collect();

    Ok(devices)
}

/// Gets an audio input device by name.
///
/// # Arguments
///
/// * `name` - The name of the device to find (case-sensitive)
///
/// # Example
///
/// ```no_run
/// use bpm_analyzer::get_device_by_name;
///
/// let device = get_device_by_name("BlackHole 2ch")?;
/// # Ok::<(), bpm_analyzer::Error>(())
/// ```
pub fn get_device_by_name(name: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();
    host.input_devices()?
        .find(|device| {
            device
                .description()
                .ok()
                .map(|desc| desc.name() == name)
                .unwrap_or(false)
        })
        .ok_or_else(|| Error::DeviceNotFound(name.to_string()))
}

/// Gets the default audio input device.
///
/// # Example
///
/// ```no_run
/// use bpm_analyzer::get_default_device;
///
/// let device = get_default_device()?;
/// # Ok::<(), bpm_analyzer::Error>(())
/// ```
pub fn get_default_device() -> Result<cpal::Device> {
    let host = cpal::default_host();
    host.default_input_device().ok_or(Error::NoDeviceFound)
}
