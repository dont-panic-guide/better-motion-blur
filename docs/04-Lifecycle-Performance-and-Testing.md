<!-- doc-type: product-implementation -->
# 04 — Lifecycle, Performance & Verification

Primary files: [CudaEngine.cpp](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/CudaEngine.cpp), [NeuralFlowEngine.cpp](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/NeuralFlowEngine.cpp), [BetterMotionBlurPlugin.cpp](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/BetterMotionBlurPlugin.cpp), [TestRunner.cpp](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/tests/TestRunner.cpp)

---

## 1. Graceful Unload & Windows Loader-Lock Safety
DaVinci Resolve loads `BetterMotionBlur.ofx` as a Windows DLL. Calling CUDA Driver teardown or joining ONNX Runtime worker threads inside `DllMain(DLL_PROCESS_DETACH)` can deadlock on the Windows OS Loader Lock or crash during `ExitProcess`:
1. **OFX `unload()` Hook (`BetterMotionBlurPluginFactory::unload`)**:
   - Invoked by Resolve (`kOfxActionUnload`) *before* `FreeLibrary`, outside the OS Loader Lock.
   - Executes in strict dependency order:
     1. `FlowCache::global().clear()` (frees cached GPU `devFlow4`/`devConf2` and `SequentialTrailState` buffers while CUDA context is alive)
     2. `CudaEngine::instance().shutdown()` (flushes stream-keyed scratch pool, unloads `CUmodule`, destroys `m_ownedCtx` if owned)
     3. `NeuralFlowEngine::instance().shutdown()` (releases `m_session`, `m_sessionOpts`, `m_cpuMemInfo`, `m_env`)
2. **Static Destruction & `ExitProcess` Guards**:
   - `CudaEngine::freeDevice()` checks `m_alive.load()` (`false` after `~CudaEngine()`) and `!isProcessShuttingDown()`.
   - `isProcessShuttingDown()` dynamically queries `RtlDllShutdownInProgress()` from `ntdll.dll`. When `TRUE`, `CudaEngine::shutdown()` and `NeuralFlowEngine::shutdown()` skip driver/ORT teardown calls and zero handles safely.
3. **Thread-Safe DLL Loading**:
   - Both `CudaEngine` and `NeuralFlowEngine` load Resolve's `nvrtc64_120_0.dll`, `nvrtc-builtins64_128.dll`, and `onnxruntime.dll` via `LoadLibraryExA(..., LOAD_WITH_ALTERED_SEARCH_PATH)` without mutating process-wide `SetDllDirectoryA`.

---

## 2. Runtime NVRTC Compilation
- No offline `nvcc.exe` is required at build time; all CUDA kernels live in `kEmbeddedCudaSource` inside `src/CudaEngine.cpp`.
- `ensureContext()` queries `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR` (`75`/`76`) and `compileKernels()` targets the native architecture (`compute_89`, `compute_86`, `compute_80`, or `compute_75`) with `--use_fast_math` and automatic `compute_75` fallback.

---

## 3. Automated Verification Suite (`tests/TestRunner.cpp`)
Run `.\build\Release\BetterMotionBlurTests.exe` after building `Release`:

| Test | What It Verifies | Passing Baseline (RTX 4090) |
| :--- | :--- | :--- |
| **Test 1** | CUDA + NVRTC & ONNX Runtime initialization | `GPU=NVIDIA GeForce RTX 4090 \| ORT=1.13.1` |
| **Test 2** | Optical flow accuracy on synthetic +12/-6 px shift | `Neural EPE=0.1082 px`, `Pyramidal EPE=0.0807 px` (`< 0.50 px`) |
| **Test 3** | Shutter blur + foreground matte isolation (DW/I) | `Core fg blue=0.0500` (no bleed), `leading edge blur red=0.7276` (`> 0.15`) |
| **Test 4** | Gapless flow-bridged trail vs discrete echo | `Discrete minR=0.0200` (gap), `Flow-Bridged minR=0.9777` (`> 0.50`) |
| **Test 5** | LRU `FlowCache` hit rate across sequential window | `hits=12, misses=2, hitRate=85.7%` (`>= 85%`) |
| **Test 6** | `BetterMotionBlur.ofx` DLL exports & plugin ID | `OfxGetNumberOfPlugins == 1`, ID `com.BetterMotionBlur.OFX.BetterMotionBlur` |
| **Test 7** | Graceful `shutdown()`, stale `freeDevice` no-op, & re-init | Exits `0` cleanly |
| **Test 8** | 1080p & 4K GPU performance + canonical cache reuse | `1080p Neural=5.1ms (197 fps, hit=50%)`, `Pyr=11.1ms (90 fps)` \| `4K Neural=10.0ms (100.5 fps)` |
| **Test 9** | Long-Exposure parity (Physical/Comet/Diag >40px fallback/Seq/Combined) & 1080p/4K cache non-thrashing | `RefPhys=0.00000`, `RefComet=0.00000`, `CpuPhys=0.0038`, `1080p8=29.0ms (hit=86%)`, `4K8=96.6ms (hit=86%)`, `4K16=99.9ms (hit=93%)` |
| **Test 10** | Tile-adaptive `autoShutterSamples` (mixed velocity, static preservation, GPU/CPU parity, `eModeCombined`) | `StaticDiff=0.00e+00`, `GpuCpuMeanDiff=0.0002` |
| **Test 11** | Physical 180° ground-truth profile, 1080p `140 px` displacement, curved exterior arc, single-sided occlusion, 5-subframe integration & `eTimebaseClipFps` | `11A RMSE G=0.0231 C=0.0354 M=0.0258`, `11B EPE N=0.04 P=0.01`, `11C +y=0.495/-y=0.040 occ=0.653`, `11D subDiff=0.0000` |
| **Test 12** | Long-Exposure curved circular arc, `eMatteForegroundIsolation` background non-smearing, short-window wake backfill & sub-frame `pairTimeStep` | `12A arcR=0.978 chordInR=0.020 diff=0.0031`, `12B smear=6.0e-08 fgR=0.608`, `12C wakeR=0.296 subDelta=0.041` |
