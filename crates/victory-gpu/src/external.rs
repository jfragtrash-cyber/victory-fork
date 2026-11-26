//! Host work that a host library records into a title's command list in
//! place of a console one (the host's FidelityFX upscaler, say): the host
//! device, a title texture's host copy in the state the library is told,
//! and the host list itself.

use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::object::Object;

/// The host D3D12 device, created on first use; `None` without a host GPU.
pub fn host_device() -> Option<ID3D12Device> {
    crate::host::host().map(|h| h.device.clone())
}

/// A title texture's host copy, moved to the host resource `state` in the
/// title list `list`, and the state it is in: `state` without the bits the
/// host copy cannot take (see `host::legal_state`). What the list records
/// next finds it there and must leave it there. The list holds the texture
/// until it is reset. `None` for a buffer, or without a host list.
///
/// The copy is first made ready as for a title shader reading it
/// ([`ready_to_read`]): the library reads it with shaders too.
pub fn host_texture(
    list: *const Object,
    resource: *const Object,
    state: u32,
) -> Option<(ID3D12Resource, u32)> {
    let host = crate::list::host_list(list)?;
    let texture = crate::resource::host_resource(resource)?.clone();
    if prepare() {
        ready_to_read(host, resource, &texture);
    }
    // What the library writes (unordered access) is the frame's too, for
    // `VICTORY_DUMP_TARGETS`.
    #[cfg(feature = "diagnostics")]
    if state & 0x8 != 0 {
        crate::capture::note_target(resource);
    }
    crate::list::ensure_state(host, resource, state);
    let state = crate::host::legal_state(&texture, state);
    Some((texture, state))
}

/// Whether [`host_texture`] makes a texture ready as a title shader's read
/// would (`VICTORY_FFX_PREPARE=0`: not, as before).
///
/// Before, it only moved the host copy to the state asked, and a host
/// library read what a title shader would not have: a view the title placed
/// in the memory of a texture the host writes was never given that
/// texture's texels (`crate::texture::alias`), a texture the title fills
/// was never uploaded from title memory (`crate::texture::upload`), and a
/// depth texture whose colour twin the same list had written was read
/// before the twin was copied back (`HostList::twin_use`; the copy waited
/// for the list's close). The upscaler read such inputs stale or empty.
fn prepare() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FFX_PREPARE").is_none_or(|v| v != "0"))
}

/// Reads by a host library of a depth texture whose depth plane lacked
/// what its twin was written in the same list: the plane is drawn from its
/// twin only when the list closes, after the library read it.
static STALE_PLANES: AtomicU64 = AtomicU64::new(0);

/// [`STALE_PLANES`] since the start.
pub fn stale_planes() -> u64 {
    STALE_PLANES.load(Ordering::Relaxed)
}

/// Brings title texture `resource` (host copy `texture`) up to date in
/// `list` for shaders to read, as `crate::draw` does for a title shader's
/// view: the texels of the texture the host wrote in the same memory, or
/// title memory, or the depth texture's colour twin.
fn ready_to_read(
    list: &crate::host::HostList,
    resource: crate::object::This,
    texture: &ID3D12Resource,
) {
    let aliased = crate::resource::alias_of(resource)
        .is_some_and(|source| crate::texture::alias(list, resource, source));
    if !aliased && crate::texture::title_written(texture) {
        crate::texture::upload(list, resource, texture);
    }
    // Counted by `twin_use` itself; the difference is ours (other threads
    // may count too: a diagnostic, not a measure).
    let late = crate::resource::PLANES[2].load(Ordering::Relaxed);
    if list.twin_use(resource, false, false)
        && crate::resource::PLANES[2].load(Ordering::Relaxed) != late
    {
        STALE_PLANES.fetch_add(1, Ordering::Relaxed);
    }
}

/// What title texture `resource` is on the host, for log lines: the host
/// copy's format, size and resource flags, and whether it has a colour
/// twin (a depth texture shaders also write: `+twin`, `+plane` if its depth
/// plane has one too), is a view placed in the memory of a texture the
/// host writes (`alias`), or is filled from title memory (`filled`).
pub fn describe(resource: *const Object) -> String {
    let Some(texture) = crate::resource::host_resource(resource) else {
        return format!("{resource:?} without a host copy");
    };
    // SAFETY: a live texture.
    let d = unsafe { texture.GetDesc() };
    let mut text = format!(
        "{resource:?} {}x{} f{} flags {:#x}",
        d.Width, d.Height, d.Format.0, d.Flags.0
    );
    if let Some(twin) = crate::resource::twin(resource) {
        text.push_str(if twin.plane.is_some() { " +twin +plane" } else { " +twin" });
    }
    if crate::resource::alias_of(resource).is_some() {
        text.push_str(" alias");
    } else if crate::texture::title_written(texture) {
        text.push_str(" filled");
    }
    text
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording); `None` without one. What `f` binds (descriptor heaps, root
/// signatures, pipeline) is replaced before the list's next draw or
/// dispatch.
pub fn record<R>(
    list: *const Object,
    f: impl FnOnce(&ID3D12GraphicsCommandList) -> R,
) -> Option<R> {
    let mut out = None;
    crate::list::with_state(list, |host, state| {
        out = Some(f(&host.list));
        state.host_state_lost();
    });
    out
}

/// The number of the frame being recorded (diagnostics).
#[cfg(feature = "diagnostics")]
pub fn frame() -> u64 {
    crate::queue::presented() + 1
}

/// Whether frame `n` is one the capture names (`VICTORY_CAPTURE_FRAMES`,
/// or a `now` file in the capture folder) (diagnostics).
#[cfg(feature = "diagnostics")]
pub fn listed(n: u64) -> bool {
    crate::capture::folder().is_some() && crate::capture::listed(n)
}

/// Adds `what` to the ops file of a listed frame, as a line of the title
/// list `list` (diagnostics).
#[cfg(feature = "diagnostics")]
pub fn note_op(list: *const Object, what: impl FnOnce() -> String) {
    if let Some(host) = crate::list::host_list(list) {
        crate::capture::op(std::ptr::from_ref(host) as usize, what);
    }
}

/// Records into the title list `list`, where it stands, a copy of title
/// texture `resource`'s host copy (in host state `state`, where it is left)
/// into readback memory, saved at a later present as
/// `frame-<frame>-<what>-...png` in the capture folder (diagnostics): what a
/// host library is handed, as the card holds it at that point.
#[cfg(feature = "diagnostics")]
pub fn snapshot(list: *const Object, resource: *const Object, state: u32, frame: u64, what: &str) {
    let (Some(host), Some(texture)) =
        (crate::list::host_list(list), crate::resource::host_resource(resource))
    else {
        return;
    };
    let state = crate::host::legal_state(texture, state);
    crate::capture::snap(host, texture, state, frame, format!("{what}-{resource:?}"));
}
