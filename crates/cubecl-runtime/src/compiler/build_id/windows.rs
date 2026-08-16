//
// - https://deplinenoise.wordpress.com/2013/06/14/getting-your-pdb-name-from-a-running-executable-windows/
// - `link.exe /DUMP /HEADERS .\target\debug\examples\simple.exe`
//    - includes the `IMAGE_DEBUG_DIRECTORY` section pretty printed

use log::error;
use winapi::um::libloaderapi::GetModuleHandleA;
use winapi::um::winnt::IMAGE_DEBUG_DIRECTORY;
use winapi::um::winnt::IMAGE_DEBUG_TYPE_CODEVIEW;
use winapi::um::winnt::{
    IMAGE_DIRECTORY_ENTRY_DEBUG, IMAGE_DOS_HEADER, IMAGE_FILE_HEADER, IMAGE_OPTIONAL_HEADER,
};

#[allow(bad_style)]
#[repr(C)]
struct CV_INFO_PDB70 {
    cv_signature: u32,
    // GUID + age
    signature_and_age: [u8; 20],
    // followed by pdb name
}

/// Same as the Windows implementation in `buildid`, but with the `Age` property included in the
/// signature. This fixes incremental builds.
pub fn build_id() -> Option<&'static [u8]> {
    let module = unsafe { GetModuleHandleA(core::ptr::null_mut()) };

    let dos_header = unsafe { &*(module as *const IMAGE_DOS_HEADER) };
    let file_header = unsafe {
        &*((module as usize + dos_header.e_lfanew as usize + 4) as *const IMAGE_FILE_HEADER)
    };

    if file_header.SizeOfOptionalHeader == 0 {
        error!("no optional header found");
        return None;
    }

    let opt_header = unsafe {
        &*((file_header as *const _ as usize + core::mem::size_of::<IMAGE_FILE_HEADER>())
            as *const IMAGE_OPTIONAL_HEADER)
    };

    if opt_header.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_DEBUG.into() {
        error!("IMAGE_DIRECTORY_ENTRY_DEBUG not included in executable");
        return None;
    }

    let dir = &opt_header.DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG as usize];
    if dir.Size == 0 {
        error!("IMAGE_DIRECTORY_ENTRY_DEBUG is empty");
        return None;
    }

    let dbg_dir = unsafe {
        &*((module as usize + dir.VirtualAddress as usize) as *const IMAGE_DEBUG_DIRECTORY)
    };

    // TODO: multiple debug directories can be present, we only examine the
    // first one which is always the one we want. We could scan all of them.
    if dbg_dir.Type != IMAGE_DEBUG_TYPE_CODEVIEW {
        error!("wrong image type {:#x}", dbg_dir.Type);
        return None;
    }

    let pdb_info = unsafe {
        &*((module as usize + dbg_dir.AddressOfRawData as usize) as *const CV_INFO_PDB70)
    };
    // 0x53445352 == "RSDS"
    if pdb_info.cv_signature != u32::from_le_bytes(*b"RSDS") {
        error!(
            "unexpected value for pdb_info cv_signature: got {:#x}",
            pdb_info.cv_signature
        );
        None
    } else {
        Some(&pdb_info.signature_and_age[..])
    }
}
