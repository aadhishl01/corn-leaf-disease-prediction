import { createRequire } from "node:module";

//#region rolldown:runtime
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __esm = (fn, res) => function() {
	return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
};
var __commonJS = (cb, mod) => function() {
	return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
};
var __export = (all) => {
	let target = {};
	for (var name in all) __defProp(target, name, {
		get: all[name],
		enumerable: true
	});
	return target;
};
var __copyProps = (to, from, except, desc) => {
	if (from && typeof from === "object" || typeof from === "function") for (var keys = __getOwnPropNames(from), i = 0, n = keys.length, key; i < n; i++) {
		key = keys[i];
		if (!__hasOwnProp.call(to, key) && key !== except) __defProp(to, key, {
			get: ((k) => from[k]).bind(null, key),
			enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable
		});
	}
	return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var __require = /* @__PURE__ */ createRequire(import.meta.url);

//#endregion
//#region ../../node_modules/.pnpm/@emnapi+wasi-threads@1.1.0/node_modules/@emnapi/wasi-threads/dist/wasi-threads.esm-bundler.js
function getPostMessage(options) {
	return typeof (options === null || options === void 0 ? void 0 : options.postMessage) === "function" ? options.postMessage : typeof postMessage === "function" ? postMessage : void 0;
}
function serizeErrorToBuffer(sab, code, error$1) {
	var i32array = new Int32Array(sab);
	Atomics.store(i32array, 0, code);
	if (code > 1 && error$1) {
		var name_1 = error$1.name;
		var message = error$1.message;
		var stack = error$1.stack;
		var nameBuffer = new TextEncoder().encode(name_1);
		var messageBuffer = new TextEncoder().encode(message);
		var stackBuffer = new TextEncoder().encode(stack);
		Atomics.store(i32array, 1, nameBuffer.length);
		Atomics.store(i32array, 2, messageBuffer.length);
		Atomics.store(i32array, 3, stackBuffer.length);
		var buffer = new Uint8Array(sab);
		buffer.set(nameBuffer, 16);
		buffer.set(messageBuffer, 16 + nameBuffer.length);
		buffer.set(stackBuffer, 16 + nameBuffer.length + messageBuffer.length);
	}
}
function deserizeErrorFromBuffer(sab) {
	var _a, _b;
	var i32array = new Int32Array(sab);
	if (Atomics.load(i32array, 0) <= 1) return null;
	var nameLength = Atomics.load(i32array, 1);
	var messageLength = Atomics.load(i32array, 2);
	var stackLength = Atomics.load(i32array, 3);
	var buffer = new Uint8Array(sab);
	var nameBuffer = buffer.slice(16, 16 + nameLength);
	var messageBuffer = buffer.slice(16 + nameLength, 16 + nameLength + messageLength);
	var stackBuffer = buffer.slice(16 + nameLength + messageLength, 16 + nameLength + messageLength + stackLength);
	var name = new TextDecoder().decode(nameBuffer);
	var message = new TextDecoder().decode(messageBuffer);
	var stack = new TextDecoder().decode(stackBuffer);
	var error$1 = new ((_a = globalThis[name]) !== null && _a !== void 0 ? _a : name === "RuntimeError" ? (_b = _WebAssembly$2.RuntimeError) !== null && _b !== void 0 ? _b : Error : Error)(message);
	Object.defineProperty(error$1, "stack", {
		value: stack,
		writable: true,
		enumerable: false,
		configurable: true
	});
	return error$1;
}
/** @public */
function isSharedArrayBuffer(value) {
	return typeof SharedArrayBuffer === "function" && value instanceof SharedArrayBuffer || Object.prototype.toString.call(value) === "[object SharedArrayBuffer]";
}
/** @public */
function isTrapError(e) {
	try {
		return e instanceof _WebAssembly$2.RuntimeError;
	} catch (_) {
		return false;
	}
}
function createMessage(type, payload) {
	return { __emnapi__: {
		type,
		payload
	} };
}
function checkSharedWasmMemory(wasmMemory) {
	if (wasmMemory) {
		if (!isSharedArrayBuffer(wasmMemory.buffer)) throw new Error("Multithread features require shared wasm memory. Try to compile with `-matomics -mbulk-memory` and use `--import-memory --shared-memory` during linking, then create WebAssembly.Memory with `shared: true` option");
	} else if (typeof SharedArrayBuffer === "undefined") throw new Error("Current environment does not support SharedArrayBuffer, threads are not available!");
}
function getReuseWorker(value) {
	var _a;
	if (typeof value === "boolean") return value ? {
		size: 0,
		strict: false
	} : false;
	if (typeof value === "number") {
		if (!(value >= 0)) throw new RangeError("reuseWorker: size must be a non-negative integer");
		return {
			size: value,
			strict: false
		};
	}
	if (!value) return false;
	var size = (_a = Number(value.size)) !== null && _a !== void 0 ? _a : 0;
	var strict = Boolean(value.strict);
	if (!(size > 0) && strict) throw new RangeError("reuseWorker: size must be set to positive integer if strict is set to true");
	return {
		size,
		strict
	};
}
/** @public */
function createInstanceProxy(instance, memory) {
	if (instance[kIsProxy]) return instance;
	var originalExports = instance.exports;
	var createHandler = function(target) {
		var handlers = [
			"apply",
			"construct",
			"defineProperty",
			"deleteProperty",
			"get",
			"getOwnPropertyDescriptor",
			"getPrototypeOf",
			"has",
			"isExtensible",
			"ownKeys",
			"preventExtensions",
			"set",
			"setPrototypeOf"
		];
		var handler$1 = {};
		var _loop_1 = function(i$1) {
			var name_1 = handlers[i$1];
			handler$1[name_1] = function() {
				var args = Array.prototype.slice.call(arguments, 1);
				args.unshift(target);
				return Reflect[name_1].apply(Reflect, args);
			};
		};
		for (var i = 0; i < handlers.length; i++) _loop_1(i);
		return handler$1;
	};
	var handler = createHandler(originalExports);
	var _initialize = function() {};
	var _start = function() {
		return 0;
	};
	handler.get = function(_target, p, receiver) {
		var _a;
		if (p === "memory") return (_a = typeof memory === "function" ? memory() : memory) !== null && _a !== void 0 ? _a : Reflect.get(originalExports, p, receiver);
		if (p === "_initialize") return p in originalExports ? _initialize : void 0;
		if (p === "_start") return p in originalExports ? _start : void 0;
		return Reflect.get(originalExports, p, receiver);
	};
	handler.has = function(_target, p) {
		if (p === "memory") return true;
		return Reflect.has(originalExports, p);
	};
	var exportsProxy = new Proxy(Object.create(null), handler);
	return new Proxy(instance, { get: function(target, p, receiver) {
		if (p === "exports") return exportsProxy;
		if (p === kIsProxy) return true;
		return Reflect.get(target, p, receiver);
	} });
}
function patchWasiInstance(wasiThreads, wasi) {
	var patched = patchedWasiInstances.get(wasiThreads);
	if (patched.has(wasi)) return;
	var _this = wasiThreads;
	var wasiImport = wasi.wasiImport;
	if (wasiImport) {
		var proc_exit_1 = wasiImport.proc_exit;
		wasiImport.proc_exit = function(code) {
			_this.terminateAllThreads();
			return proc_exit_1.call(this, code);
		};
	}
	if (!_this.childThread) {
		var start_1 = wasi.start;
		if (typeof start_1 === "function") wasi.start = function(instance) {
			try {
				return start_1.call(this, instance);
			} catch (err) {
				if (isTrapError(err)) _this.terminateAllThreads();
				throw err;
			}
		};
	}
	patched.add(wasi);
}
function getWasiSymbol(wasi, description) {
	var symbols = Object.getOwnPropertySymbols(wasi);
	var selectDescription = function(description$1) {
		return function(s) {
			if (s.description) return s.description === description$1;
			return s.toString() === "Symbol(".concat(description$1, ")");
		};
	};
	if (Array.isArray(description)) return description.map(function(d) {
		return symbols.filter(selectDescription(d))[0];
	});
	return symbols.filter(selectDescription(description))[0];
}
function setupInstance(wasi, instance) {
	var _a = getWasiSymbol(wasi, ["kInstance", "kSetMemory"]), kInstance$1 = _a[0], kSetMemory$1 = _a[1];
	wasi[kInstance$1] = instance;
	wasi[kSetMemory$1](instance.exports.memory);
}
function notifyPthreadCreateResult(sab, result, error$1) {
	if (sab) {
		serizeErrorToBuffer(sab.buffer, result, error$1);
		Atomics.notify(sab, 0);
	}
}
var _WebAssembly$2, ENVIRONMENT_IS_NODE, WASI_THREADS_MAX_TID, nextWorkerID, ThreadManager, kIsProxy, patchedWasiInstances, WASIThreads, ThreadMessageHandler;
var init_wasi_threads_esm_bundler = __esm({ "../../node_modules/.pnpm/@emnapi+wasi-threads@1.1.0/node_modules/@emnapi/wasi-threads/dist/wasi-threads.esm-bundler.js": (() => {
	_WebAssembly$2 = typeof WebAssembly !== "undefined" ? WebAssembly : typeof WXWebAssembly !== "undefined" ? WXWebAssembly : void 0;
	ENVIRONMENT_IS_NODE = typeof process === "object" && process !== null && typeof process.versions === "object" && process.versions !== null && typeof process.versions.node === "string";
	WASI_THREADS_MAX_TID = 536870911;
	nextWorkerID = 0;
	ThreadManager = /* @__PURE__ */ function() {
		function ThreadManager$1(options) {
			var _a;
			this.unusedWorkers = [];
			this.runningWorkers = [];
			this.pthreads = Object.create(null);
			this.wasmModule = null;
			this.wasmMemory = null;
			this.messageEvents = /* @__PURE__ */ new WeakMap();
			if (!options) throw new TypeError("ThreadManager(): options is not provided");
			if ("childThread" in options) this._childThread = Boolean(options.childThread);
			else this._childThread = false;
			if (this._childThread) {
				this._onCreateWorker = void 0;
				this._reuseWorker = false;
				this._beforeLoad = void 0;
			} else {
				this._onCreateWorker = options.onCreateWorker;
				this._reuseWorker = getReuseWorker(options.reuseWorker);
				this._beforeLoad = options.beforeLoad;
			}
			this.printErr = (_a = options.printErr) !== null && _a !== void 0 ? _a : console.error.bind(console);
		}
		Object.defineProperty(ThreadManager$1.prototype, "nextWorkerID", {
			get: function() {
				return nextWorkerID;
			},
			enumerable: false,
			configurable: true
		});
		ThreadManager$1.prototype.init = function() {
			if (!this._childThread) this.initMainThread();
		};
		ThreadManager$1.prototype.initMainThread = function() {
			this.preparePool();
		};
		ThreadManager$1.prototype.preparePool = function() {
			if (this._reuseWorker) {
				if (this._reuseWorker.size) {
					var pthreadPoolSize = this._reuseWorker.size;
					while (pthreadPoolSize--) {
						var worker = this.allocateUnusedWorker();
						if (ENVIRONMENT_IS_NODE) {
							worker.once("message", function() {});
							worker.unref();
						}
					}
				}
			}
		};
		ThreadManager$1.prototype.shouldPreloadWorkers = function() {
			return !this._childThread && this._reuseWorker && this._reuseWorker.size > 0;
		};
		ThreadManager$1.prototype.loadWasmModuleToAllWorkers = function() {
			var _this_1 = this;
			var promises = Array(this.unusedWorkers.length);
			var _loop_1 = function(i$1) {
				var worker = this_1.unusedWorkers[i$1];
				if (ENVIRONMENT_IS_NODE) worker.ref();
				promises[i$1] = this_1.loadWasmModuleToWorker(worker).then(function(w) {
					if (ENVIRONMENT_IS_NODE) worker.unref();
					return w;
				}, function(e) {
					if (ENVIRONMENT_IS_NODE) worker.unref();
					throw e;
				});
			};
			var this_1 = this;
			for (var i = 0; i < this.unusedWorkers.length; ++i) _loop_1(i);
			return Promise.all(promises).catch(function(err) {
				_this_1.terminateAllThreads();
				throw err;
			});
		};
		ThreadManager$1.prototype.preloadWorkers = function() {
			if (this.shouldPreloadWorkers()) return this.loadWasmModuleToAllWorkers();
			return Promise.resolve([]);
		};
		ThreadManager$1.prototype.setup = function(wasmModule, wasmMemory) {
			this.wasmModule = wasmModule;
			this.wasmMemory = wasmMemory;
		};
		ThreadManager$1.prototype.markId = function(worker) {
			if (worker.__emnapi_tid) return worker.__emnapi_tid;
			var tid = nextWorkerID + 43;
			nextWorkerID = (nextWorkerID + 1) % (WASI_THREADS_MAX_TID - 42);
			this.pthreads[tid] = worker;
			worker.__emnapi_tid = tid;
			return tid;
		};
		ThreadManager$1.prototype.returnWorkerToPool = function(worker) {
			var tid = worker.__emnapi_tid;
			if (tid !== void 0) delete this.pthreads[tid];
			this.unusedWorkers.push(worker);
			this.runningWorkers.splice(this.runningWorkers.indexOf(worker), 1);
			delete worker.__emnapi_tid;
			if (ENVIRONMENT_IS_NODE) worker.unref();
		};
		ThreadManager$1.prototype.loadWasmModuleToWorker = function(worker, sab) {
			var _this_1 = this;
			if (worker.whenLoaded) return worker.whenLoaded;
			var err = this.printErr;
			var beforeLoad = this._beforeLoad;
			var _this = this;
			worker.whenLoaded = new Promise(function(resolve$1, reject) {
				var handleError$1 = function(e) {
					var message = "worker sent an error!";
					if (worker.__emnapi_tid !== void 0) message = "worker (tid = " + worker.__emnapi_tid + ") sent an error!";
					if ("message" in e) {
						err(message + " " + e.message);
						if (e.message.indexOf("RuntimeError") !== -1 || e.message.indexOf("unreachable") !== -1) try {
							_this.terminateAllThreads();
						} catch (_) {}
					} else err(message);
					reject(e);
					throw e;
				};
				var handleMessage = function(data) {
					if (data.__emnapi__) {
						var type = data.__emnapi__.type;
						var payload = data.__emnapi__.payload;
						if (type === "loaded") {
							worker.loaded = true;
							if (ENVIRONMENT_IS_NODE && !worker.__emnapi_tid) worker.unref();
							resolve$1(worker);
						} else if (type === "cleanup-thread") {
							if (payload.tid in _this_1.pthreads) _this_1.cleanThread(worker, payload.tid);
						}
					}
				};
				worker.onmessage = function(e) {
					handleMessage(e.data);
					_this_1.fireMessageEvent(worker, e);
				};
				worker.onerror = handleError$1;
				if (ENVIRONMENT_IS_NODE) {
					worker.on("message", function(data) {
						var _a, _b;
						(_b = (_a = worker).onmessage) === null || _b === void 0 || _b.call(_a, { data });
					});
					worker.on("error", function(e) {
						var _a, _b;
						(_b = (_a = worker).onerror) === null || _b === void 0 || _b.call(_a, e);
					});
					worker.on("detachedExit", function() {});
				}
				if (typeof beforeLoad === "function") beforeLoad(worker);
				try {
					worker.postMessage(createMessage("load", {
						wasmModule: _this_1.wasmModule,
						wasmMemory: _this_1.wasmMemory,
						sab
					}));
				} catch (err$1) {
					checkSharedWasmMemory(_this_1.wasmMemory);
					throw err$1;
				}
			});
			return worker.whenLoaded;
		};
		ThreadManager$1.prototype.allocateUnusedWorker = function() {
			var _onCreateWorker = this._onCreateWorker;
			if (typeof _onCreateWorker !== "function") throw new TypeError("`options.onCreateWorker` is not provided");
			var worker = _onCreateWorker({
				type: "thread",
				name: "emnapi-pthread"
			});
			this.unusedWorkers.push(worker);
			return worker;
		};
		ThreadManager$1.prototype.getNewWorker = function(sab) {
			if (this._reuseWorker) {
				if (this.unusedWorkers.length === 0) {
					if (this._reuseWorker.strict) {
						if (!ENVIRONMENT_IS_NODE) {
							var err = this.printErr;
							err("Tried to spawn a new thread, but the thread pool is exhausted.\nThis might result in a deadlock unless some threads eventually exit or the code explicitly breaks out to the event loop.");
							return;
						}
					}
					var worker_1 = this.allocateUnusedWorker();
					this.loadWasmModuleToWorker(worker_1, sab);
				}
				return this.unusedWorkers.pop();
			}
			var worker = this.allocateUnusedWorker();
			this.loadWasmModuleToWorker(worker, sab);
			return this.unusedWorkers.pop();
		};
		ThreadManager$1.prototype.cleanThread = function(worker, tid, force) {
			if (!force && this._reuseWorker) this.returnWorkerToPool(worker);
			else {
				delete this.pthreads[tid];
				var index = this.runningWorkers.indexOf(worker);
				if (index !== -1) this.runningWorkers.splice(index, 1);
				this.terminateWorker(worker);
				delete worker.__emnapi_tid;
			}
		};
		ThreadManager$1.prototype.terminateWorker = function(worker) {
			var _this_1 = this;
			var _a;
			var tid = worker.__emnapi_tid;
			worker.terminate();
			(_a = this.messageEvents.get(worker)) === null || _a === void 0 || _a.clear();
			this.messageEvents.delete(worker);
			worker.onmessage = function(e) {
				if (e.data.__emnapi__) {
					var err = _this_1.printErr;
					err("received \"" + e.data.__emnapi__.type + "\" command from terminated worker: " + tid);
				}
			};
		};
		ThreadManager$1.prototype.terminateAllThreads = function() {
			for (var i = 0; i < this.runningWorkers.length; ++i) this.terminateWorker(this.runningWorkers[i]);
			for (var i = 0; i < this.unusedWorkers.length; ++i) this.terminateWorker(this.unusedWorkers[i]);
			this.unusedWorkers = [];
			this.runningWorkers = [];
			this.pthreads = Object.create(null);
			this.preparePool();
		};
		ThreadManager$1.prototype.addMessageEventListener = function(worker, onMessage) {
			var listeners = this.messageEvents.get(worker);
			if (!listeners) {
				listeners = /* @__PURE__ */ new Set();
				this.messageEvents.set(worker, listeners);
			}
			listeners.add(onMessage);
			return function() {
				listeners === null || listeners === void 0 || listeners.delete(onMessage);
			};
		};
		ThreadManager$1.prototype.fireMessageEvent = function(worker, e) {
			var listeners = this.messageEvents.get(worker);
			if (!listeners) return;
			var err = this.printErr;
			listeners.forEach(function(listener) {
				try {
					listener(e);
				} catch (e$1) {
					err(e$1.stack);
				}
			});
		};
		return ThreadManager$1;
	}();
	kIsProxy = Symbol("kIsProxy");
	patchedWasiInstances = /* @__PURE__ */ new WeakMap();
	WASIThreads = /* @__PURE__ */ function() {
		function WASIThreads$1(options) {
			var _this_1 = this;
			if (!options) throw new TypeError("WASIThreads(): options is not provided");
			if (!options.wasi) throw new TypeError("WASIThreads(): options.wasi is not provided");
			patchedWasiInstances.set(this, /* @__PURE__ */ new WeakSet());
			var wasi = options.wasi;
			patchWasiInstance(this, wasi);
			this.wasi = wasi;
			if ("childThread" in options) this.childThread = Boolean(options.childThread);
			else this.childThread = false;
			this.PThread = void 0;
			if ("threadManager" in options) if (typeof options.threadManager === "function") this.PThread = options.threadManager();
			else this.PThread = options.threadManager;
			else if (!this.childThread) {
				this.PThread = new ThreadManager(options);
				this.PThread.init();
			}
			var waitThreadStart = false;
			if ("waitThreadStart" in options) waitThreadStart = typeof options.waitThreadStart === "number" ? options.waitThreadStart : Boolean(options.waitThreadStart);
			var postMessage$1 = getPostMessage(options);
			if (this.childThread && typeof postMessage$1 !== "function") throw new TypeError("options.postMessage is not a function");
			this.postMessage = postMessage$1;
			var wasm64 = Boolean(options.wasm64);
			var onMessage = function(e) {
				if (e.data.__emnapi__) {
					var type = e.data.__emnapi__.type;
					var payload = e.data.__emnapi__.payload;
					if (type === "spawn-thread") threadSpawn(payload.startArg, payload.errorOrTid);
					else if (type === "terminate-all-threads") _this_1.terminateAllThreads();
				}
			};
			var threadSpawn = function(startArg, errorOrTid) {
				var _a;
				var EAGAIN = 6;
				var isNewABI = errorOrTid !== void 0;
				try {
					checkSharedWasmMemory(_this_1.wasmMemory);
				} catch (err) {
					(_a = _this_1.PThread) === null || _a === void 0 || _a.printErr(err.stack);
					if (isNewABI) {
						var struct_1 = new Int32Array(_this_1.wasmMemory.buffer, errorOrTid, 2);
						Atomics.store(struct_1, 0, 1);
						Atomics.store(struct_1, 1, EAGAIN);
						Atomics.notify(struct_1, 1);
						return 1;
					} else return -EAGAIN;
				}
				if (!isNewABI) {
					var malloc = _this_1.wasmInstance.exports.malloc;
					errorOrTid = wasm64 ? Number(malloc(BigInt(8))) : malloc(8);
					if (!errorOrTid) return -48;
				}
				var _free = _this_1.wasmInstance.exports.free;
				var free = wasm64 ? function(ptr) {
					_free(BigInt(ptr));
				} : _free;
				var struct = new Int32Array(_this_1.wasmMemory.buffer, errorOrTid, 2);
				Atomics.store(struct, 0, 0);
				Atomics.store(struct, 1, 0);
				if (_this_1.childThread) {
					postMessage$1(createMessage("spawn-thread", {
						startArg,
						errorOrTid
					}));
					Atomics.wait(struct, 1, 0);
					var isError = Atomics.load(struct, 0);
					var result = Atomics.load(struct, 1);
					if (isNewABI) return isError;
					free(errorOrTid);
					return isError ? -result : result;
				}
				var shouldWait = waitThreadStart || waitThreadStart === 0;
				var sab;
				if (shouldWait) {
					sab = new Int32Array(new SharedArrayBuffer(8208));
					Atomics.store(sab, 0, 0);
				}
				var worker;
				var tid;
				var PThread = _this_1.PThread;
				try {
					worker = PThread.getNewWorker(sab);
					if (!worker) throw new Error("failed to get new worker");
					PThread.addMessageEventListener(worker, onMessage);
					tid = PThread.markId(worker);
					if (ENVIRONMENT_IS_NODE) worker.ref();
					worker.postMessage(createMessage("start", {
						tid,
						arg: startArg,
						sab
					}));
					if (shouldWait) {
						if (typeof waitThreadStart === "number") {
							if (Atomics.wait(sab, 0, 0, waitThreadStart) === "timed-out") {
								try {
									PThread.cleanThread(worker, tid, true);
								} catch (_) {}
								throw new Error("Spawning thread timed out. Please check if the worker is created successfully and if message is handled properly in the worker.");
							}
						} else Atomics.wait(sab, 0, 0);
						if (Atomics.load(sab, 0) > 1) {
							try {
								PThread.cleanThread(worker, tid, true);
							} catch (_) {}
							throw deserizeErrorFromBuffer(sab.buffer);
						}
					}
				} catch (e) {
					Atomics.store(struct, 0, 1);
					Atomics.store(struct, 1, EAGAIN);
					Atomics.notify(struct, 1);
					PThread === null || PThread === void 0 || PThread.printErr(e.stack);
					if (isNewABI) return 1;
					free(errorOrTid);
					return -EAGAIN;
				}
				Atomics.store(struct, 0, 0);
				Atomics.store(struct, 1, tid);
				Atomics.notify(struct, 1);
				PThread.runningWorkers.push(worker);
				if (!shouldWait) worker.whenLoaded.catch(function(err) {
					delete worker.whenLoaded;
					PThread.cleanThread(worker, tid, true);
					throw err;
				});
				if (isNewABI) return 0;
				free(errorOrTid);
				return tid;
			};
			this.threadSpawn = threadSpawn;
		}
		WASIThreads$1.prototype.getImportObject = function() {
			return { wasi: { "thread-spawn": this.threadSpawn } };
		};
		WASIThreads$1.prototype.setup = function(wasmInstance, wasmModule, wasmMemory) {
			wasmMemory !== null && wasmMemory !== void 0 || (wasmMemory = wasmInstance.exports.memory);
			this.wasmInstance = wasmInstance;
			this.wasmMemory = wasmMemory;
			if (this.PThread) this.PThread.setup(wasmModule, wasmMemory);
		};
		WASIThreads$1.prototype.preloadWorkers = function() {
			if (this.PThread) return this.PThread.preloadWorkers();
			return Promise.resolve([]);
		};
		/**
		* It's ok to call this method to a WASI command module.
		*
		* in child thread, must call this method instead of {@link WASIThreads.start} even if it's a WASI command module
		*
		* @returns A proxied WebAssembly instance if in child thread, other wise the original instance
		*/
		WASIThreads$1.prototype.initialize = function(instance, module$1, memory) {
			var exports$1 = instance.exports;
			memory !== null && memory !== void 0 || (memory = exports$1.memory);
			if (this.childThread) instance = createInstanceProxy(instance, memory);
			this.setup(instance, module$1, memory);
			var wasi = this.wasi;
			if ("_start" in exports$1 && typeof exports$1._start === "function") if (this.childThread) {
				wasi.start(instance);
				try {
					var kStarted$1 = getWasiSymbol(wasi, "kStarted");
					wasi[kStarted$1] = false;
				} catch (_) {}
			} else setupInstance(wasi, instance);
			else wasi.initialize(instance);
			return instance;
		};
		/**
		* Equivalent to calling {@link WASIThreads.initialize} and then calling {@link WASIInstance.start}
		* ```js
		* this.initialize(instance, module, memory)
		* this.wasi.start(instance)
		* ```
		*/
		WASIThreads$1.prototype.start = function(instance, module$1, memory) {
			var exports$1 = instance.exports;
			memory !== null && memory !== void 0 || (memory = exports$1.memory);
			if (this.childThread) instance = createInstanceProxy(instance, memory);
			this.setup(instance, module$1, memory);
			return {
				exitCode: this.wasi.start(instance),
				instance
			};
		};
		WASIThreads$1.prototype.terminateAllThreads = function() {
			var _a;
			if (!this.childThread) (_a = this.PThread) === null || _a === void 0 || _a.terminateAllThreads();
			else this.postMessage(createMessage("terminate-all-threads", {}));
		};
		return WASIThreads$1;
	}();
	ThreadMessageHandler = /* @__PURE__ */ function() {
		function ThreadMessageHandler$1(options) {
			var postMsg = getPostMessage(options);
			if (typeof postMsg !== "function") throw new TypeError("options.postMessage is not a function");
			this.postMessage = postMsg;
			this.onLoad = options === null || options === void 0 ? void 0 : options.onLoad;
			this.onError = typeof (options === null || options === void 0 ? void 0 : options.onError) === "function" ? options.onError : function(_type, err) {
				throw err;
			};
			this.instance = void 0;
			this.messagesBeforeLoad = [];
		}
		/** @virtual */
		ThreadMessageHandler$1.prototype.instantiate = function(data) {
			if (typeof this.onLoad === "function") return this.onLoad(data);
			throw new Error("ThreadMessageHandler.prototype.instantiate is not implemented");
		};
		/** @virtual */
		ThreadMessageHandler$1.prototype.handle = function(e) {
			var _this = this;
			var _a;
			if ((_a = e === null || e === void 0 ? void 0 : e.data) === null || _a === void 0 ? void 0 : _a.__emnapi__) {
				var type = e.data.__emnapi__.type;
				var payload_1 = e.data.__emnapi__.payload;
				try {
					if (type === "load") this._load(payload_1);
					else if (type === "start") this.handleAfterLoad(e, function() {
						_this._start(payload_1);
					});
				} catch (err) {
					this.onError(err, type);
				}
			}
		};
		ThreadMessageHandler$1.prototype._load = function(payload) {
			var _this = this;
			if (this.instance !== void 0) return;
			var source;
			try {
				source = this.instantiate(payload);
			} catch (err) {
				this._loaded(err, null, payload);
				return;
			}
			var then = source && "then" in source ? source.then : void 0;
			if (typeof then === "function") then.call(source, function(source$1) {
				_this._loaded(null, source$1, payload);
			}, function(err) {
				_this._loaded(err, null, payload);
			});
			else this._loaded(null, source, payload);
		};
		ThreadMessageHandler$1.prototype._start = function(payload) {
			var wasi_thread_start = this.instance.exports.wasi_thread_start;
			if (typeof wasi_thread_start !== "function") {
				var err = /* @__PURE__ */ new TypeError("wasi_thread_start is not exported");
				notifyPthreadCreateResult(payload.sab, 2, err);
				throw err;
			}
			var postMessage$1 = this.postMessage;
			var tid = payload.tid;
			var startArg = payload.arg;
			notifyPthreadCreateResult(payload.sab, 1);
			try {
				wasi_thread_start(tid, startArg);
			} catch (err$1) {
				if (err$1 !== "unwind") throw err$1;
				else return;
			}
			postMessage$1(createMessage("cleanup-thread", { tid }));
		};
		ThreadMessageHandler$1.prototype._loaded = function(err, source, payload) {
			if (err) {
				notifyPthreadCreateResult(payload.sab, 2, err);
				throw err;
			}
			if (source == null) {
				var err_1 = /* @__PURE__ */ new TypeError("onLoad should return an object");
				notifyPthreadCreateResult(payload.sab, 2, err_1);
				throw err_1;
			}
			var instance = source.instance;
			if (!instance) {
				var err_2 = /* @__PURE__ */ new TypeError("onLoad should return an object which includes \"instance\"");
				notifyPthreadCreateResult(payload.sab, 2, err_2);
				throw err_2;
			}
			this.instance = instance;
			var postMessage$1 = this.postMessage;
			postMessage$1(createMessage("loaded", {}));
			var messages = this.messagesBeforeLoad;
			this.messagesBeforeLoad = [];
			for (var i = 0; i < messages.length; i++) {
				var data = messages[i];
				this.handle({ data });
			}
		};
		ThreadMessageHandler$1.prototype.handleAfterLoad = function(e, f) {
			if (this.instance !== void 0) f.call(this, e);
			else this.messagesBeforeLoad.push(e.data);
		};
		return ThreadMessageHandler$1;
	}();
}) });

//#endregion
//#region ../../node_modules/.pnpm/tslib@2.8.1/node_modules/tslib/tslib.es6.mjs
function __extends(d, b) {
	if (typeof b !== "function" && b !== null) throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
	extendStatics(d, b);
	function __() {
		this.constructor = d;
	}
	d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}
var extendStatics, __assign;
var init_tslib_es6 = __esm({ "../../node_modules/.pnpm/tslib@2.8.1/node_modules/tslib/tslib.es6.mjs": (() => {
	extendStatics = function(d, b) {
		extendStatics = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(d$1, b$1) {
			d$1.__proto__ = b$1;
		} || function(d$1, b$1) {
			for (var p in b$1) if (Object.prototype.hasOwnProperty.call(b$1, p)) d$1[p] = b$1[p];
		};
		return extendStatics(d, b);
	};
	__assign = function() {
		__assign = Object.assign || function __assign$1(t) {
			for (var s, i = 1, n = arguments.length; i < n; i++) {
				s = arguments[i];
				for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p)) t[p] = s[p];
			}
			return t;
		};
		return __assign.apply(this, arguments);
	};
}) });

//#endregion
//#region ../../node_modules/.pnpm/@emnapi+core@1.5.0/node_modules/@emnapi/core/dist/emnapi-core.esm-bundler.js
var emnapi_core_esm_bundler_exports = /* @__PURE__ */ __export({
	MessageHandler: () => MessageHandler$1,
	ThreadManager: () => ThreadManager,
	ThreadMessageHandler: () => ThreadMessageHandler,
	WASIThreads: () => WASIThreads,
	createInstanceProxy: () => createInstanceProxy,
	createNapiModule: () => createNapiModule,
	instantiateNapiModule: () => instantiateNapiModule$1,
	instantiateNapiModuleSync: () => instantiateNapiModuleSync$1,
	isSharedArrayBuffer: () => isSharedArrayBuffer,
	isTrapError: () => isTrapError,
	loadNapiModule: () => loadNapiModule,
	loadNapiModuleSync: () => loadNapiModuleSync,
	version: () => version$2
});
function validateImports$1(imports) {
	if (imports && typeof imports !== "object") throw new TypeError("imports must be an object or undefined");
	return true;
}
function load$1(wasmInput, imports) {
	if (!wasmInput) throw new TypeError("Invalid wasm source");
	validateImports$1(imports);
	imports = imports !== null && imports !== void 0 ? imports : {};
	try {
		var then = typeof wasmInput === "object" && wasmInput !== null && "then" in wasmInput ? wasmInput.then : void 0;
		if (typeof then === "function") return then.call(wasmInput, function(input) {
			return load$1(input, imports);
		});
	} catch (_) {}
	if (wasmInput instanceof ArrayBuffer || ArrayBuffer.isView(wasmInput)) return _WebAssembly$1.instantiate(wasmInput, imports);
	if (wasmInput instanceof _WebAssembly$1.Module) return _WebAssembly$1.instantiate(wasmInput, imports).then(function(instance) {
		return {
			instance,
			module: wasmInput
		};
	});
	if (typeof Response !== "undefined" && wasmInput instanceof Response) return wasmInput.arrayBuffer().then(function(buffer) {
		return _WebAssembly$1.instantiate(buffer, imports);
	});
	var inputIsString = typeof wasmInput === "string";
	if (inputIsString || typeof URL !== "undefined" && wasmInput instanceof URL) {
		if (inputIsString && typeof wx !== "undefined" && typeof __wxConfig !== "undefined") return _WebAssembly$1.instantiate(wasmInput, imports);
		if (typeof fetch !== "function") throw new TypeError("wasm source can not be a string or URL in this environment");
		if (typeof _WebAssembly$1.instantiateStreaming === "function") try {
			return _WebAssembly$1.instantiateStreaming(fetch(wasmInput), imports).catch(function() {
				return load$1(fetch(wasmInput), imports);
			});
		} catch (_) {
			return load$1(fetch(wasmInput), imports);
		}
		else return load$1(fetch(wasmInput), imports);
	}
	throw new TypeError("Invalid wasm source");
}
function loadSync$1(wasmInput, imports) {
	if (!wasmInput) throw new TypeError("Invalid wasm source");
	validateImports$1(imports);
	imports = imports !== null && imports !== void 0 ? imports : {};
	var module$1;
	if (wasmInput instanceof ArrayBuffer || ArrayBuffer.isView(wasmInput)) module$1 = new _WebAssembly$1.Module(wasmInput);
	else if (wasmInput instanceof WebAssembly.Module) module$1 = wasmInput;
	else throw new TypeError("Invalid wasm source");
	return {
		instance: new _WebAssembly$1.Instance(module$1, imports),
		module: module$1
	};
}
function createNapiModule(options) {
	return (function() {
		var ENVIRONMENT_IS_NODE$1 = typeof process === "object" && process !== null && typeof process.versions === "object" && process.versions !== null && typeof process.versions.node === "string";
		var ENVIRONMENT_IS_PTHREAD = Boolean(options.childThread);
		var waitThreadStart = typeof options.waitThreadStart === "number" ? options.waitThreadStart : Boolean(options.waitThreadStart);
		var wasmInstance;
		var wasmMemory;
		var wasmTable;
		var _malloc;
		var _free;
		function abort(msg) {
			if (typeof _WebAssembly$1.RuntimeError === "function") throw new _WebAssembly$1.RuntimeError(msg);
			throw Error(msg);
		}
		var napiModule = {
			imports: {
				env: {},
				napi: {},
				emnapi: {}
			},
			exports: {},
			emnapi: {},
			loaded: false,
			filename: "",
			childThread: ENVIRONMENT_IS_PTHREAD,
			initWorker: void 0,
			executeAsyncWork: void 0,
			waitThreadStart,
			PThread: void 0,
			init: function(options$1) {
				if (napiModule.loaded) return napiModule.exports;
				if (!options$1) throw new TypeError("Invalid napi init options");
				var instance = options$1.instance;
				if (!(instance === null || instance === void 0 ? void 0 : instance.exports)) throw new TypeError("Invalid wasm instance");
				wasmInstance = instance;
				var exports$1 = instance.exports;
				var module$1 = options$1.module;
				var memory = options$1.memory || exports$1.memory;
				var table = options$1.table || exports$1.__indirect_function_table;
				if (!(module$1 instanceof _WebAssembly$1.Module)) throw new TypeError("Invalid wasm module");
				if (!(memory instanceof _WebAssembly$1.Memory)) throw new TypeError("Invalid wasm memory");
				if (!(table instanceof _WebAssembly$1.Table)) throw new TypeError("Invalid wasm table");
				wasmMemory = memory;
				wasmTable = table;
				if (typeof exports$1.malloc !== "function") throw new TypeError("malloc is not exported");
				if (typeof exports$1.free !== "function") throw new TypeError("free is not exported");
				_malloc = exports$1.malloc;
				_free = exports$1.free;
				if (!napiModule.childThread) {
					var moduleApiVersion = 8;
					var node_api_module_get_api_version_v1 = instance.exports.node_api_module_get_api_version_v1;
					if (typeof node_api_module_get_api_version_v1 === "function") moduleApiVersion = node_api_module_get_api_version_v1();
					var envObject = napiModule.envObject || (napiModule.envObject = emnapiCtx.createEnv(napiModule.filename, moduleApiVersion, function(cb) {
						return wasmTable.get(cb);
					}, function(cb) {
						return wasmTable.get(cb);
					}, abort, emnapiNodeBinding));
					var scope_1 = emnapiCtx.openScope(envObject);
					try {
						envObject.callIntoModule(function(_envObject) {
							var exports$2 = napiModule.exports;
							var exportsHandle = scope_1.add(exports$2);
							var napi_register_wasm_v1 = instance.exports.napi_register_wasm_v1;
							var napiValue = napi_register_wasm_v1(_envObject.id, exportsHandle.id);
							napiModule.exports = !napiValue ? exports$2 : emnapiCtx.handleStore.get(napiValue).value;
						});
					} catch (e) {
						if (e !== "unwind") throw e;
					} finally {
						emnapiCtx.closeScope(envObject, scope_1);
					}
					napiModule.loaded = true;
					delete napiModule.envObject;
					return napiModule.exports;
				}
			}
		};
		var emnapiCtx;
		var emnapiNodeBinding;
		var onCreateWorker = void 0;
		var err;
		if (!ENVIRONMENT_IS_PTHREAD) {
			var context = options.context;
			if (typeof context !== "object" || context === null) throw new TypeError("Invalid `options.context`. Use `import { getDefaultContext } from '@emnapi/runtime'`");
			emnapiCtx = context;
		} else {
			emnapiCtx = options === null || options === void 0 ? void 0 : options.context;
			var postMsg = typeof options.postMessage === "function" ? options.postMessage : typeof postMessage === "function" ? postMessage : void 0;
			if (typeof postMsg !== "function") throw new TypeError("No postMessage found");
			napiModule.postMessage = postMsg;
		}
		if (typeof options.filename === "string") napiModule.filename = options.filename;
		if (typeof options.onCreateWorker === "function") onCreateWorker = options.onCreateWorker;
		if (typeof options.print === "function") options.print;
		else console.log.bind(console);
		if (typeof options.printErr === "function") err = options.printErr;
		else err = console.warn.bind(console);
		if ("nodeBinding" in options) {
			var nodeBinding = options.nodeBinding;
			if (typeof nodeBinding !== "object" || nodeBinding === null) throw new TypeError("Invalid `options.nodeBinding`. Use @emnapi/node-binding package");
			emnapiNodeBinding = nodeBinding;
		}
		var emnapiAsyncWorkPoolSize = 0;
		if ("asyncWorkPoolSize" in options) {
			if (typeof options.asyncWorkPoolSize !== "number") throw new TypeError("options.asyncWorkPoolSize must be a integer");
			emnapiAsyncWorkPoolSize = options.asyncWorkPoolSize >> 0;
			if (emnapiAsyncWorkPoolSize > 1024) emnapiAsyncWorkPoolSize = 1024;
			else if (emnapiAsyncWorkPoolSize < -1024) emnapiAsyncWorkPoolSize = -1024;
		}
		var singleThreadAsyncWork = ENVIRONMENT_IS_PTHREAD ? false : emnapiAsyncWorkPoolSize <= 0;
		function _emnapi_async_work_pool_size() {
			return Math.abs(emnapiAsyncWorkPoolSize);
		}
		napiModule.imports.env._emnapi_async_work_pool_size = _emnapi_async_work_pool_size;
		function emnapiAddSendListener(worker) {
			if (!worker) return false;
			if (worker._emnapiSendListener) return true;
			var handler = function(e) {
				var __emnapi__ = (ENVIRONMENT_IS_NODE$1 ? e : e.data).__emnapi__;
				if (__emnapi__ && __emnapi__.type === "async-send") if (ENVIRONMENT_IS_PTHREAD) {
					var postMessage_1 = napiModule.postMessage;
					postMessage_1({ __emnapi__ });
				} else {
					var callback = __emnapi__.payload.callback;
					wasmTable.get(callback)(__emnapi__.payload.data);
				}
			};
			var dispose = function() {
				if (ENVIRONMENT_IS_NODE$1) worker.off("message", handler);
				else worker.removeEventListener("message", handler, false);
				delete worker._emnapiSendListener;
			};
			worker._emnapiSendListener = {
				handler,
				dispose
			};
			if (ENVIRONMENT_IS_NODE$1) worker.on("message", handler);
			else worker.addEventListener("message", handler, false);
			return true;
		}
		napiModule.emnapi.addSendListener = emnapiAddSendListener;
		var PThread = new ThreadManager(ENVIRONMENT_IS_PTHREAD ? {
			printErr: err,
			childThread: true
		} : {
			printErr: err,
			beforeLoad: function(worker) {
				emnapiAddSendListener(worker);
			},
			reuseWorker: options.reuseWorker,
			onCreateWorker
		});
		napiModule.PThread = PThread;
		/**
		* @__sig ipiip
		*/
		function napi_set_last_error(env, error_code, engine_error_code, engine_reserved) {
			return emnapiCtx.envStore.get(env).setLastError(error_code, engine_error_code, engine_reserved);
		}
		/**
		* @__sig ip
		*/
		function napi_clear_last_error(env) {
			return emnapiCtx.envStore.get(env).clearLastError();
		}
		/**
		* @__sig vppp
		*/
		function _emnapi_get_node_version(major, minor, patch) {
			var versions = typeof process === "object" && process !== null && typeof process.versions === "object" && process.versions !== null && typeof process.versions.node === "string" ? process.versions.node.split(".").map(function(n) {
				return Number(n);
			}) : [
				0,
				0,
				0
			];
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			HEAP_DATA_VIEW.setUint32(major, versions[0], true);
			HEAP_DATA_VIEW.setUint32(minor, versions[1], true);
			HEAP_DATA_VIEW.setUint32(patch, versions[2], true);
		}
		/**
		* @__sig v
		* @__deps $runtimeKeepalivePush
		*/
		function _emnapi_runtime_keepalive_push() {}
		/**
		* @__sig v
		* @__deps $runtimeKeepalivePop
		*/
		function _emnapi_runtime_keepalive_pop() {}
		/**
		* @__sig vpp
		*/
		function _emnapi_set_immediate(callback, data) {
			emnapiCtx.feature.setImmediate(function() {
				wasmTable.get(callback)(data);
			});
		}
		/**
		* @__sig vpp
		*/
		function _emnapi_next_tick(callback, data) {
			Promise.resolve().then(function() {
				wasmTable.get(callback)(data);
			});
		}
		/**
		* @__sig vipppi
		*/
		function _emnapi_callback_into_module(forceUncaught, env, callback, data, close_scope_if_throw) {
			var envObject = emnapiCtx.envStore.get(env);
			var scope = emnapiCtx.openScope(envObject);
			try {
				envObject.callbackIntoModule(Boolean(forceUncaught), function() {
					wasmTable.get(callback)(env, data);
				});
			} catch (err$1) {
				emnapiCtx.closeScope(envObject, scope);
				if (close_scope_if_throw) emnapiCtx.closeScope(envObject);
				throw err$1;
			}
			emnapiCtx.closeScope(envObject, scope);
		}
		/**
		* @__sig vipppp
		*/
		function _emnapi_call_finalizer(forceUncaught, env, callback, data, hint) {
			emnapiCtx.envStore.get(env).callFinalizerInternal(forceUncaught, callback, data, hint);
		}
		/**
		* @__sig v
		*/
		function _emnapi_ctx_increase_waiting_request_counter() {
			emnapiCtx.increaseWaitingRequestCounter();
		}
		/**
		* @__sig v
		*/
		function _emnapi_ctx_decrease_waiting_request_counter() {
			emnapiCtx.decreaseWaitingRequestCounter();
		}
		/**
		* @__sig i
		*/
		function _emnapi_is_main_runtime_thread() {
			return ENVIRONMENT_IS_PTHREAD ? 0 : 1;
		}
		/**
		* @__sig i
		*/
		function _emnapi_is_main_browser_thread() {
			return typeof window !== "undefined" && typeof document !== "undefined" && !ENVIRONMENT_IS_NODE$1 ? 1 : 0;
		}
		/**
		* @__sig v
		*/
		function _emnapi_unwind() {
			throw "unwind";
		}
		/**
		* @__sig d
		*/
		function _emnapi_get_now() {
			return performance.timeOrigin + performance.now();
		}
		function $emnapiSetValueI64(result, numberValue) {
			var tempDouble;
			var tempI64 = [numberValue >>> 0, (tempDouble = numberValue, +Math.abs(tempDouble) >= 1 ? tempDouble > 0 ? (Math.min(+Math.floor(tempDouble / 4294967296), 4294967295) | 0) >>> 0 : ~~+Math.ceil((tempDouble - +(~~tempDouble >>> 0)) / 4294967296) >>> 0 : 0)];
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			HEAP_DATA_VIEW.setInt32(result, tempI64[0], true);
			HEAP_DATA_VIEW.setInt32(result + 4, tempI64[1], true);
		}
		/**
		* @__deps $emnapiCtx
		* @__sig p
		*/
		function _emnapi_open_handle_scope() {
			return emnapiCtx.openScope().id;
		}
		/**
		* @__deps $emnapiCtx
		* @__sig vp
		*/
		function _emnapi_close_handle_scope(_scope) {
			return emnapiCtx.closeScope();
		}
		var utilMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			$emnapiSetValueI64,
			_emnapi_call_finalizer,
			_emnapi_callback_into_module,
			_emnapi_close_handle_scope,
			_emnapi_ctx_decrease_waiting_request_counter,
			_emnapi_ctx_increase_waiting_request_counter,
			_emnapi_get_node_version,
			_emnapi_get_now,
			_emnapi_is_main_browser_thread,
			_emnapi_is_main_runtime_thread,
			_emnapi_next_tick,
			_emnapi_open_handle_scope,
			_emnapi_runtime_keepalive_pop,
			_emnapi_runtime_keepalive_push,
			_emnapi_set_immediate,
			_emnapi_unwind,
			napi_clear_last_error,
			napi_set_last_error
		});
		function emnapiGetWorkerByPthreadPtr(pthreadPtr) {
			var tid = new DataView(wasmMemory.buffer).getInt32(pthreadPtr + 20, true);
			return PThread.pthreads[tid];
		}
		/** @__sig vp */
		function _emnapi_worker_unref(pthreadPtr) {
			if (ENVIRONMENT_IS_PTHREAD) return;
			var worker = emnapiGetWorkerByPthreadPtr(pthreadPtr);
			if (worker && typeof worker.unref === "function") worker.unref();
		}
		/** @__sig vipp */
		function _emnapi_async_send_js(type, callback, data) {
			if (ENVIRONMENT_IS_PTHREAD) {
				var postMessage_1 = napiModule.postMessage;
				postMessage_1({ __emnapi__: {
					type: "async-send",
					payload: {
						callback,
						data
					}
				} });
			} else switch (type) {
				case 0:
					_emnapi_set_immediate(callback, data);
					break;
				case 1:
					_emnapi_next_tick(callback, data);
					break;
			}
		}
		var uvThreadpoolReadyResolve;
		var uvThreadpoolReady = new Promise(function(resolve$1) {
			uvThreadpoolReadyResolve = function() {
				uvThreadpoolReady.ready = true;
				resolve$1();
			};
		});
		uvThreadpoolReady.ready = false;
		/** @__sig vppi */
		function _emnapi_after_uvthreadpool_ready(callback, q, type) {
			if (uvThreadpoolReady.ready) wasmTable.get(callback)(q, type);
			else uvThreadpoolReady.then(function() {
				wasmTable.get(callback)(q, type);
			});
		}
		/** @__sig vpi */
		function _emnapi_tell_js_uvthreadpool(threads, size) {
			var p = [];
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			var _loop_1 = function(i$1) {
				var pthreadPtr = HEAP_DATA_VIEW.getUint32(threads + i$1 * 4, true);
				var worker = emnapiGetWorkerByPthreadPtr(pthreadPtr);
				p.push(new Promise(function(resolve$1) {
					var handler = function(e) {
						var __emnapi__ = (ENVIRONMENT_IS_NODE$1 ? e : e.data).__emnapi__;
						if (__emnapi__ && __emnapi__.type === "async-thread-ready") {
							resolve$1();
							if (worker && typeof worker.unref === "function") worker.unref();
							if (ENVIRONMENT_IS_NODE$1) worker.off("message", handler);
							else worker.removeEventListener("message", handler);
						}
					};
					if (ENVIRONMENT_IS_NODE$1) worker.on("message", handler);
					else worker.addEventListener("message", handler);
				}));
			};
			for (var i = 0; i < size; i++) _loop_1(i);
			Promise.all(p).then(uvThreadpoolReadyResolve);
		}
		/** @__sig v */
		function _emnapi_emit_async_thread_ready() {
			if (!ENVIRONMENT_IS_PTHREAD) return;
			var postMessage$1 = napiModule.postMessage;
			postMessage$1({ __emnapi__: {
				type: "async-thread-ready",
				payload: {}
			} });
		}
		var asyncMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			_emnapi_after_uvthreadpool_ready,
			_emnapi_async_send_js,
			_emnapi_emit_async_thread_ready,
			_emnapi_tell_js_uvthreadpool,
			_emnapi_worker_unref
		});
		/** @__sig ipjp */
		function napi_adjust_external_memory(env, change_in_bytes, adjusted_value) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!adjusted_value) return envObject.setLastError(1);
			var change_in_bytes_number = Number(change_in_bytes);
			if (change_in_bytes_number < 0) return envObject.setLastError(1);
			var old_size = wasmMemory.buffer.byteLength;
			var new_size = old_size + change_in_bytes_number;
			new_size = new_size + (65536 - new_size % 65536) % 65536;
			if (wasmMemory.grow(new_size - old_size + 65535 >> 16) === -1) return envObject.setLastError(9);
			if (emnapiCtx.feature.supportBigInt) new DataView(wasmMemory.buffer).setBigInt64(adjusted_value, BigInt(wasmMemory.buffer.byteLength), true);
			else $emnapiSetValueI64(adjusted_value, wasmMemory.buffer.byteLength);
			return envObject.clearLastError();
		}
		var memoryMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_adjust_external_memory
		});
		/**
		* @__postset
		* ```
		* emnapiAWST.init();
		* ```
		*/
		var emnapiAWST = {
			idGen: {},
			values: [void 0],
			queued: /* @__PURE__ */ new Set(),
			pending: [],
			init: function() {
				var idGen = {
					nextId: 1,
					list: [],
					generate: function() {
						var id;
						if (idGen.list.length) id = idGen.list.shift();
						else {
							id = idGen.nextId;
							idGen.nextId++;
						}
						return id;
					},
					reuse: function(id) {
						idGen.list.push(id);
					}
				};
				emnapiAWST.idGen = idGen;
				emnapiAWST.values = [void 0];
				emnapiAWST.queued = /* @__PURE__ */ new Set();
				emnapiAWST.pending = [];
			},
			create: function(env, resource, resourceName, execute, complete, data) {
				var asyncId = 0;
				var triggerAsyncId = 0;
				if (emnapiNodeBinding) {
					var asyncContext = emnapiNodeBinding.node.emitAsyncInit(resource, resourceName, -1);
					asyncId = asyncContext.asyncId;
					triggerAsyncId = asyncContext.triggerAsyncId;
				}
				var id = emnapiAWST.idGen.generate();
				emnapiAWST.values[id] = {
					env,
					id,
					resource,
					asyncId,
					triggerAsyncId,
					status: 0,
					execute,
					complete,
					data
				};
				return id;
			},
			callComplete: function(work, status) {
				var complete = work.complete;
				var env = work.env;
				var data = work.data;
				var callback = function() {
					if (!complete) return;
					var envObject = emnapiCtx.envStore.get(env);
					var scope = emnapiCtx.openScope(envObject);
					try {
						envObject.callbackIntoModule(true, function() {
							wasmTable.get(complete)(env, status, data);
						});
					} finally {
						emnapiCtx.closeScope(envObject, scope);
					}
				};
				if (emnapiNodeBinding) emnapiNodeBinding.node.makeCallback(work.resource, callback, [], {
					asyncId: work.asyncId,
					triggerAsyncId: work.triggerAsyncId
				});
				else callback();
			},
			queue: function(id) {
				var work = emnapiAWST.values[id];
				if (!work) return;
				if (work.status === 0) {
					work.status = 1;
					if (emnapiAWST.queued.size >= (Math.abs(emnapiAsyncWorkPoolSize) || 4)) {
						emnapiAWST.pending.push(id);
						return;
					}
					emnapiAWST.queued.add(id);
					var env_1 = work.env;
					var data_1 = work.data;
					var execute = work.execute;
					work.status = 2;
					emnapiCtx.feature.setImmediate(function() {
						wasmTable.get(execute)(env_1, data_1);
						emnapiAWST.queued.delete(id);
						work.status = 3;
						emnapiCtx.feature.setImmediate(function() {
							emnapiAWST.callComplete(work, 0);
						});
						if (emnapiAWST.pending.length > 0) {
							var nextWorkId = emnapiAWST.pending.shift();
							emnapiAWST.values[nextWorkId].status = 0;
							emnapiAWST.queue(nextWorkId);
						}
					});
				}
			},
			cancel: function(id) {
				var index = emnapiAWST.pending.indexOf(id);
				if (index !== -1) {
					var work_1 = emnapiAWST.values[id];
					if (work_1 && work_1.status === 1) {
						work_1.status = 4;
						emnapiAWST.pending.splice(index, 1);
						emnapiCtx.feature.setImmediate(function() {
							emnapiAWST.callComplete(work_1, 11);
						});
						return 0;
					} else return 9;
				}
				return 9;
			},
			remove: function(id) {
				var work = emnapiAWST.values[id];
				if (!work) return;
				if (emnapiNodeBinding) emnapiNodeBinding.node.emitAsyncDestroy({
					asyncId: work.asyncId,
					triggerAsyncId: work.triggerAsyncId
				});
				emnapiAWST.values[id] = void 0;
				emnapiAWST.idGen.reuse(id);
			}
		};
		/** @__sig vppdp */
		function _emnapi_node_emit_async_init(async_resource, async_resource_name, trigger_async_id, result) {
			if (!emnapiNodeBinding) return;
			var resource = emnapiCtx.handleStore.get(async_resource).value;
			var resource_name = emnapiCtx.handleStore.get(async_resource_name).value;
			var asyncContext = emnapiNodeBinding.node.emitAsyncInit(resource, resource_name, trigger_async_id);
			var asyncId = asyncContext.asyncId;
			var triggerAsyncId = asyncContext.triggerAsyncId;
			if (result) {
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				HEAP_DATA_VIEW.setFloat64(result, asyncId, true);
				HEAP_DATA_VIEW.setFloat64(result + 8, triggerAsyncId, true);
			}
		}
		/** @__sig vdd */
		function _emnapi_node_emit_async_destroy(async_id, trigger_async_id) {
			if (!emnapiNodeBinding) return;
			emnapiNodeBinding.node.emitAsyncDestroy({
				asyncId: async_id,
				triggerAsyncId: trigger_async_id
			});
		}
		/** @__sig ipppppddp */
		function _emnapi_node_make_callback(env, async_resource, cb, argv, size, async_id, trigger_async_id, result) {
			var i = 0;
			var v;
			if (!emnapiNodeBinding) return;
			var resource = emnapiCtx.handleStore.get(async_resource).value;
			var callback = emnapiCtx.handleStore.get(cb).value;
			size = size >>> 0;
			var arr = Array(size);
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			for (; i < size; i++) {
				var argVal = HEAP_DATA_VIEW.getUint32(argv + i * 4, true);
				arr[i] = emnapiCtx.handleStore.get(argVal).value;
			}
			var ret = emnapiNodeBinding.node.makeCallback(resource, callback, arr, {
				asyncId: async_id,
				triggerAsyncId: trigger_async_id
			});
			if (result) {
				v = emnapiCtx.envStore.get(env).ensureHandleId(ret);
				HEAP_DATA_VIEW.setUint32(result, v, true);
			}
		}
		/** @__sig ippp */
		function _emnapi_async_init_js(async_resource, async_resource_name, result) {
			if (!emnapiNodeBinding) return 9;
			var resource;
			if (async_resource) resource = Object(emnapiCtx.handleStore.get(async_resource).value);
			var name = emnapiCtx.handleStore.get(async_resource_name).value;
			var ret = emnapiNodeBinding.napi.asyncInit(resource, name);
			if (ret.status !== 0) return ret.status;
			var numberValue = ret.value;
			if (!(numberValue >= BigInt(-1) * (BigInt(1) << BigInt(63)) && numberValue < BigInt(1) << BigInt(63))) {
				numberValue = numberValue & (BigInt(1) << BigInt(64)) - BigInt(1);
				if (numberValue >= BigInt(1) << BigInt(63)) numberValue = numberValue - (BigInt(1) << BigInt(64));
			}
			var low = Number(numberValue & BigInt(4294967295));
			var high = Number(numberValue >> BigInt(32));
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			HEAP_DATA_VIEW.setInt32(result, low, true);
			HEAP_DATA_VIEW.setInt32(result + 4, high, true);
			return 0;
		}
		/** @__sig ip */
		function _emnapi_async_destroy_js(async_context) {
			if (!emnapiNodeBinding) return 9;
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			var low = HEAP_DATA_VIEW.getInt32(async_context, true);
			var high = HEAP_DATA_VIEW.getInt32(async_context + 4, true);
			var pointer = BigInt(low >>> 0) | BigInt(high) << BigInt(32);
			var ret = emnapiNodeBinding.napi.asyncDestroy(pointer);
			if (ret.status !== 0) return ret.status;
			return 0;
		}
		/** @__sig ipppp */
		function napi_open_callback_scope(env, ignored, async_context_handle, result) {
			throw new Error("napi_open_callback_scope has not been implemented yet");
		}
		/** @__sig ipp */
		function napi_close_callback_scope(env, scope) {
			throw new Error("napi_close_callback_scope has not been implemented yet");
		}
		/** @__sig ippppppp */
		function napi_make_callback(env, async_context, recv, func, argc, argv, result) {
			var i = 0;
			var v;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!emnapiNodeBinding) return envObject.setLastError(9);
				if (!recv) return envObject.setLastError(1);
				if (argc > 0) {
					if (!argv) return envObject.setLastError(1);
				}
				var v8recv = Object(emnapiCtx.handleStore.get(recv).value);
				var v8func = emnapiCtx.handleStore.get(func).value;
				if (typeof v8func !== "function") return envObject.setLastError(1);
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				var low = HEAP_DATA_VIEW.getInt32(async_context, true);
				var high = HEAP_DATA_VIEW.getInt32(async_context + 4, true);
				var ctx = BigInt(low >>> 0) | BigInt(high) << BigInt(32);
				argc = argc >>> 0;
				var arr = Array(argc);
				for (; i < argc; i++) {
					var argVal = HEAP_DATA_VIEW.getUint32(argv + i * 4, true);
					arr[i] = emnapiCtx.handleStore.get(argVal).value;
				}
				var ret = emnapiNodeBinding.napi.makeCallback(ctx, v8recv, v8func, arr);
				if (ret.error) throw ret.error;
				if (ret.status !== 0) return envObject.setLastError(ret.status);
				if (result) {
					v = envObject.ensureHandleId(ret.value);
					HEAP_DATA_VIEW.setUint32(result, v, true);
				}
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig vp */
		function _emnapi_env_check_gc_access(env) {
			emnapiCtx.envStore.get(env).checkGCAccess();
		}
		var nodeMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			_emnapi_async_destroy_js,
			_emnapi_async_init_js,
			_emnapi_env_check_gc_access,
			_emnapi_node_emit_async_destroy,
			_emnapi_node_emit_async_init,
			_emnapi_node_make_callback,
			napi_close_callback_scope,
			napi_make_callback,
			napi_open_callback_scope
		});
		/**
		* @__deps malloc
		* @__deps free
		* @__postset
		* ```
		* emnapiTSFN.init();
		* ```
		*/
		var emnapiTSFN = {
			offset: {
				resource: 0,
				async_id: 8,
				trigger_async_id: 16,
				queue_size: 24,
				queue: 28,
				thread_count: 32,
				is_closing: 36,
				dispatch_state: 40,
				context: 44,
				max_queue_size: 48,
				ref: 52,
				env: 56,
				finalize_data: 60,
				finalize_cb: 64,
				call_js_cb: 68,
				handles_closing: 72,
				async_ref: 76,
				mutex: 80,
				cond: 84,
				end: 88
			},
			init: function() {
				if (typeof PThread !== "undefined") {
					PThread.unusedWorkers.forEach(emnapiTSFN.addListener);
					PThread.runningWorkers.forEach(emnapiTSFN.addListener);
					var __original_getNewWorker_1 = PThread.getNewWorker;
					PThread.getNewWorker = function() {
						var r = __original_getNewWorker_1.apply(this, arguments);
						emnapiTSFN.addListener(r);
						return r;
					};
				}
			},
			addListener: function(worker) {
				if (!worker) return false;
				if (worker._emnapiTSFNListener) return true;
				var handler = function(e) {
					var __emnapi__ = (ENVIRONMENT_IS_NODE$1 ? e : e.data).__emnapi__;
					if (__emnapi__) {
						var type = __emnapi__.type;
						var payload = __emnapi__.payload;
						if (type === "tsfn-send") emnapiTSFN.dispatch(payload.tsfn);
					}
				};
				var dispose = function() {
					if (ENVIRONMENT_IS_NODE$1) worker.off("message", handler);
					else worker.removeEventListener("message", handler, false);
					delete worker._emnapiTSFNListener;
				};
				worker._emnapiTSFNListener = {
					handler,
					dispose
				};
				if (ENVIRONMENT_IS_NODE$1) worker.on("message", handler);
				else worker.addEventListener("message", handler, false);
				return true;
			},
			initQueue: function(func) {
				var size = 8;
				var queue = _malloc(size);
				if (!queue) return false;
				new Uint8Array(wasmMemory.buffer, queue, size).fill(0);
				emnapiTSFN.storeSizeTypeValue(func + emnapiTSFN.offset.queue, queue, false);
				return true;
			},
			destroyQueue: function(func) {
				var queue = emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.queue, false);
				if (queue) _free(queue);
			},
			pushQueue: function(func, data) {
				var queue = emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.queue, false);
				var head = emnapiTSFN.loadSizeTypeValue(queue, false);
				var tail = emnapiTSFN.loadSizeTypeValue(queue + 4, false);
				var node = _malloc(8);
				if (!node) throw new Error("OOM");
				emnapiTSFN.storeSizeTypeValue(node, data, false);
				emnapiTSFN.storeSizeTypeValue(node + 4, 0, false);
				if (head === 0 && tail === 0) {
					emnapiTSFN.storeSizeTypeValue(queue, node, false);
					emnapiTSFN.storeSizeTypeValue(queue + 4, node, false);
				} else {
					emnapiTSFN.storeSizeTypeValue(tail + 4, node, false);
					emnapiTSFN.storeSizeTypeValue(queue + 4, node, false);
				}
				emnapiTSFN.addQueueSize(func);
			},
			shiftQueue: function(func) {
				var queue = emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.queue, false);
				var head = emnapiTSFN.loadSizeTypeValue(queue, false);
				if (head === 0) return 0;
				var node = head;
				var next = emnapiTSFN.loadSizeTypeValue(head + 4, false);
				emnapiTSFN.storeSizeTypeValue(queue, next, false);
				if (next === 0) emnapiTSFN.storeSizeTypeValue(queue + 4, 0, false);
				emnapiTSFN.storeSizeTypeValue(node + 4, 0, false);
				var value = emnapiTSFN.loadSizeTypeValue(node, false);
				_free(node);
				emnapiTSFN.subQueueSize(func);
				return value;
			},
			push: function(func, data, mode) {
				var mutex = emnapiTSFN.getMutex(func);
				var cond = emnapiTSFN.getCond(func);
				var waitCondition = function() {
					var queueSize = emnapiTSFN.getQueueSize(func);
					var maxSize = emnapiTSFN.getMaxQueueSize(func);
					var isClosing = emnapiTSFN.getIsClosing(func);
					return queueSize >= maxSize && maxSize > 0 && !isClosing;
				};
				var isBrowserMain = typeof window !== "undefined" && typeof document !== "undefined" && !ENVIRONMENT_IS_NODE$1;
				return mutex.execute(function() {
					while (waitCondition()) {
						if (mode === 0) return 15;
						/**
						* Browser JS main thread can not use `Atomics.wait`
						*
						* Related:
						* https://github.com/nodejs/node/pull/32689
						* https://github.com/nodejs/node/pull/33453
						*/
						if (isBrowserMain) return 21;
						cond.wait();
					}
					if (emnapiTSFN.getIsClosing(func)) if (emnapiTSFN.getThreadCount(func) === 0) return 1;
					else {
						emnapiTSFN.subThreadCount(func);
						return 16;
					}
					else {
						emnapiTSFN.pushQueue(func, data);
						emnapiTSFN.send(func);
						return 0;
					}
				});
			},
			getMutex: function(func) {
				var index = func + emnapiTSFN.offset.mutex;
				var mutex = {
					lock: function() {
						var isBrowserMain = typeof window !== "undefined" && typeof document !== "undefined" && !ENVIRONMENT_IS_NODE$1;
						var i32a = new Int32Array(wasmMemory.buffer, index, 1);
						if (isBrowserMain) while (true) {
							var oldValue = Atomics.compareExchange(i32a, 0, 0, 1);
							if (oldValue === 0) return;
						}
						else while (true) {
							var oldValue = Atomics.compareExchange(i32a, 0, 0, 1);
							if (oldValue === 0) return;
							Atomics.wait(i32a, 0, 1);
						}
					},
					unlock: function() {
						var i32a = new Int32Array(wasmMemory.buffer, index, 1);
						if (Atomics.compareExchange(i32a, 0, 1, 0) !== 1) throw new Error("Tried to unlock while not holding the mutex");
						Atomics.notify(i32a, 0, 1);
					},
					execute: function(fn) {
						mutex.lock();
						try {
							return fn();
						} finally {
							mutex.unlock();
						}
					}
				};
				return mutex;
			},
			getCond: function(func) {
				var index = func + emnapiTSFN.offset.cond;
				var mutex = emnapiTSFN.getMutex(func);
				return {
					wait: function() {
						var i32a = new Int32Array(wasmMemory.buffer, index, 1);
						var value = Atomics.load(i32a, 0);
						mutex.unlock();
						Atomics.wait(i32a, 0, value);
						mutex.lock();
					},
					signal: function() {
						var i32a = new Int32Array(wasmMemory.buffer, index, 1);
						Atomics.add(i32a, 0, 1);
						Atomics.notify(i32a, 0, 1);
					}
				};
			},
			getQueueSize: function(func) {
				return emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.queue_size, true);
			},
			addQueueSize: function(func) {
				var offset = emnapiTSFN.offset.queue_size;
				var arr = new Uint32Array(wasmMemory.buffer), index = func + offset >> 2;
				Atomics.add(arr, index, 1);
			},
			subQueueSize: function(func) {
				var offset = emnapiTSFN.offset.queue_size;
				var arr = new Uint32Array(wasmMemory.buffer), index = func + offset >> 2;
				Atomics.sub(arr, index, 1);
			},
			getThreadCount: function(func) {
				return emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.thread_count, true);
			},
			addThreadCount: function(func) {
				var offset = emnapiTSFN.offset.thread_count;
				var arr = new Uint32Array(wasmMemory.buffer), index = func + offset >> 2;
				Atomics.add(arr, index, 1);
			},
			subThreadCount: function(func) {
				var offset = emnapiTSFN.offset.thread_count;
				var arr = new Uint32Array(wasmMemory.buffer), index = func + offset >> 2;
				Atomics.sub(arr, index, 1);
			},
			getIsClosing: function(func) {
				return Atomics.load(new Int32Array(wasmMemory.buffer), func + emnapiTSFN.offset.is_closing >> 2);
			},
			setIsClosing: function(func, value) {
				Atomics.store(new Int32Array(wasmMemory.buffer), func + emnapiTSFN.offset.is_closing >> 2, value);
			},
			getHandlesClosing: function(func) {
				return Atomics.load(new Int32Array(wasmMemory.buffer), func + emnapiTSFN.offset.handles_closing >> 2);
			},
			setHandlesClosing: function(func, value) {
				Atomics.store(new Int32Array(wasmMemory.buffer), func + emnapiTSFN.offset.handles_closing >> 2, value);
			},
			getDispatchState: function(func) {
				return Atomics.load(new Uint32Array(wasmMemory.buffer), func + emnapiTSFN.offset.dispatch_state >> 2);
			},
			getContext: function(func) {
				return emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.context, false);
			},
			getMaxQueueSize: function(func) {
				return emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.max_queue_size, true);
			},
			getEnv: function(func) {
				return emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.env, false);
			},
			getCallJSCb: function(func) {
				return emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.call_js_cb, false);
			},
			getRef: function(func) {
				return emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.ref, false);
			},
			getResource: function(func) {
				return emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.resource, false);
			},
			getFinalizeCb: function(func) {
				return emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.finalize_cb, false);
			},
			getFinalizeData: function(func) {
				return emnapiTSFN.loadSizeTypeValue(func + emnapiTSFN.offset.finalize_data, false);
			},
			loadSizeTypeValue: function(offset, unsigned) {
				var ret;
				var arr;
				if (unsigned) {
					arr = new Uint32Array(wasmMemory.buffer);
					ret = Atomics.load(arr, offset >> 2);
					return ret;
				} else {
					arr = new Int32Array(wasmMemory.buffer);
					ret = Atomics.load(arr, offset >> 2);
					return ret;
				}
			},
			storeSizeTypeValue: function(offset, value, unsigned) {
				var arr;
				if (unsigned) {
					arr = new Uint32Array(wasmMemory.buffer);
					Atomics.store(arr, offset >> 2, value);
					return;
				} else {
					arr = new Int32Array(wasmMemory.buffer);
					Atomics.store(arr, offset >> 2, value >>> 0);
					return;
				}
			},
			destroy: function(func) {
				emnapiTSFN.destroyQueue(func);
				var env = emnapiTSFN.getEnv(func);
				var envObject = emnapiCtx.envStore.get(env);
				var ref = emnapiTSFN.getRef(func);
				if (ref) emnapiCtx.refStore.get(ref).dispose();
				emnapiCtx.removeCleanupHook(envObject, emnapiTSFN.cleanup, func);
				envObject.unref();
				var asyncRefOffset = func + emnapiTSFN.offset.async_ref >> 2;
				var arr = new Int32Array(wasmMemory.buffer);
				if (Atomics.load(arr, asyncRefOffset)) {
					Atomics.store(arr, asyncRefOffset, 0);
					emnapiCtx.decreaseWaitingRequestCounter();
				}
				var resource = emnapiTSFN.getResource(func);
				emnapiCtx.refStore.get(resource).dispose();
				if (emnapiNodeBinding) {
					var view = new DataView(wasmMemory.buffer);
					var asyncId = view.getFloat64(func + emnapiTSFN.offset.async_id, true);
					var triggerAsyncId = view.getFloat64(func + emnapiTSFN.offset.trigger_async_id, true);
					_emnapi_node_emit_async_destroy(asyncId, triggerAsyncId);
				}
				_free(func);
			},
			emptyQueueAndDelete: function(func) {
				var callJsCb = emnapiTSFN.getCallJSCb(func);
				var context$1 = emnapiTSFN.getContext(func);
				var data;
				while (emnapiTSFN.getQueueSize(func) > 0) {
					data = emnapiTSFN.shiftQueue(func);
					if (callJsCb) wasmTable.get(callJsCb)(0, 0, context$1, data);
				}
				emnapiTSFN.destroy(func);
			},
			finalize: function(func) {
				var env = emnapiTSFN.getEnv(func);
				var envObject = emnapiCtx.envStore.get(env);
				emnapiCtx.openScope(envObject);
				var finalize = emnapiTSFN.getFinalizeCb(func);
				var data = emnapiTSFN.getFinalizeData(func);
				var context$1 = emnapiTSFN.getContext(func);
				var f = function() {
					envObject.callFinalizerInternal(0, finalize, data, context$1);
				};
				try {
					if (finalize) if (emnapiNodeBinding) {
						var resource = emnapiTSFN.getResource(func);
						var resource_value = emnapiCtx.refStore.get(resource).get();
						var resourceObject = emnapiCtx.handleStore.get(resource_value).value;
						var view = new DataView(wasmMemory.buffer);
						var asyncId = view.getFloat64(func + emnapiTSFN.offset.async_id, true);
						var triggerAsyncId = view.getFloat64(func + emnapiTSFN.offset.trigger_async_id, true);
						emnapiNodeBinding.node.makeCallback(resourceObject, f, [], {
							asyncId,
							triggerAsyncId
						});
					} else f();
					emnapiTSFN.emptyQueueAndDelete(func);
				} finally {
					emnapiCtx.closeScope(envObject);
				}
			},
			cleanup: function(func) {
				emnapiTSFN.closeHandlesAndMaybeDelete(func, 1);
			},
			closeHandlesAndMaybeDelete: function(func, set_closing) {
				var env = emnapiTSFN.getEnv(func);
				var envObject = emnapiCtx.envStore.get(env);
				emnapiCtx.openScope(envObject);
				try {
					if (set_closing) emnapiTSFN.getMutex(func).execute(function() {
						emnapiTSFN.setIsClosing(func, 1);
						if (emnapiTSFN.getMaxQueueSize(func) > 0) emnapiTSFN.getCond(func).signal();
					});
					if (emnapiTSFN.getHandlesClosing(func)) return;
					emnapiTSFN.setHandlesClosing(func, 1);
					emnapiCtx.feature.setImmediate(function() {
						emnapiTSFN.finalize(func);
					});
				} finally {
					emnapiCtx.closeScope(envObject);
				}
			},
			dispatchOne: function(func) {
				var data = 0;
				var popped_value = false;
				var has_more = false;
				var mutex = emnapiTSFN.getMutex(func);
				var cond = emnapiTSFN.getCond(func);
				mutex.execute(function() {
					if (emnapiTSFN.getIsClosing(func)) emnapiTSFN.closeHandlesAndMaybeDelete(func, 0);
					else {
						var size = emnapiTSFN.getQueueSize(func);
						if (size > 0) {
							data = emnapiTSFN.shiftQueue(func);
							popped_value = true;
							var maxQueueSize = emnapiTSFN.getMaxQueueSize(func);
							if (size === maxQueueSize && maxQueueSize > 0) cond.signal();
							size--;
						}
						if (size === 0) {
							if (emnapiTSFN.getThreadCount(func) === 0) {
								emnapiTSFN.setIsClosing(func, 1);
								if (emnapiTSFN.getMaxQueueSize(func) > 0) cond.signal();
								emnapiTSFN.closeHandlesAndMaybeDelete(func, 0);
							}
						} else has_more = true;
					}
				});
				if (popped_value) {
					var env = emnapiTSFN.getEnv(func);
					var envObject_1 = emnapiCtx.envStore.get(env);
					emnapiCtx.openScope(envObject_1);
					var f = function() {
						envObject_1.callbackIntoModule(false, function() {
							var callJsCb = emnapiTSFN.getCallJSCb(func);
							var ref = emnapiTSFN.getRef(func);
							var js_callback = ref ? emnapiCtx.refStore.get(ref).get() : 0;
							if (callJsCb) {
								var context$1 = emnapiTSFN.getContext(func);
								wasmTable.get(callJsCb)(env, js_callback, context$1, data);
							} else {
								var jsCallback = js_callback ? emnapiCtx.handleStore.get(js_callback).value : null;
								if (typeof jsCallback === "function") jsCallback();
							}
						});
					};
					try {
						if (emnapiNodeBinding) {
							var resource = emnapiTSFN.getResource(func);
							var resource_value = emnapiCtx.refStore.get(resource).get();
							var resourceObject = emnapiCtx.handleStore.get(resource_value).value;
							var view = new DataView(wasmMemory.buffer);
							emnapiNodeBinding.node.makeCallback(resourceObject, f, [], {
								asyncId: view.getFloat64(func + emnapiTSFN.offset.async_id, true),
								triggerAsyncId: view.getFloat64(func + emnapiTSFN.offset.trigger_async_id, true)
							});
						} else f();
					} finally {
						emnapiCtx.closeScope(envObject_1);
					}
				}
				return has_more;
			},
			dispatch: function(func) {
				var has_more = true;
				var iterations_left = 1e3;
				var ui32a = new Uint32Array(wasmMemory.buffer);
				var index = func + emnapiTSFN.offset.dispatch_state >> 2;
				while (has_more && --iterations_left !== 0) {
					Atomics.store(ui32a, index, 1);
					has_more = emnapiTSFN.dispatchOne(func);
					if (Atomics.exchange(ui32a, index, 0) !== 1) has_more = true;
				}
				if (has_more) emnapiTSFN.send(func);
			},
			send: function(func) {
				if ((Atomics.or(new Uint32Array(wasmMemory.buffer), func + emnapiTSFN.offset.dispatch_state >> 2, 2) & 1) === 1) return;
				if (typeof ENVIRONMENT_IS_PTHREAD !== "undefined" && ENVIRONMENT_IS_PTHREAD) postMessage({ __emnapi__: {
					type: "tsfn-send",
					payload: { tsfn: func }
				} });
				else emnapiCtx.feature.setImmediate(function() {
					emnapiTSFN.dispatch(func);
				});
			}
		};
		/** @__sig ippppppppppp */
		function napi_create_threadsafe_function(env, func, async_resource, async_resource_name, max_queue_size, initial_thread_count, thread_finalize_data, thread_finalize_cb, context$1, call_js_cb, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!async_resource_name) return envObject.setLastError(1);
			max_queue_size = max_queue_size >>> 0;
			initial_thread_count = initial_thread_count >>> 0;
			if (initial_thread_count === 0) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var ref = 0;
			if (!func) {
				if (!call_js_cb) return envObject.setLastError(1);
			} else {
				if (typeof emnapiCtx.handleStore.get(func).value !== "function") return envObject.setLastError(1);
				ref = emnapiCtx.createReference(envObject, func, 1, 1).id;
			}
			var asyncResourceObject;
			if (async_resource) {
				asyncResourceObject = emnapiCtx.handleStore.get(async_resource).value;
				if (asyncResourceObject == null) return envObject.setLastError(2);
				asyncResourceObject = Object(asyncResourceObject);
			} else asyncResourceObject = {};
			var resource = envObject.ensureHandleId(asyncResourceObject);
			var asyncResourceName = emnapiCtx.handleStore.get(async_resource_name).value;
			if (typeof asyncResourceName === "symbol") return envObject.setLastError(3);
			asyncResourceName = String(asyncResourceName);
			var resource_name = envObject.ensureHandleId(asyncResourceName);
			var sizeofTSFN = emnapiTSFN.offset.end;
			var tsfn = _malloc(sizeofTSFN);
			if (!tsfn) return envObject.setLastError(9);
			new Uint8Array(wasmMemory.buffer).subarray(tsfn, tsfn + sizeofTSFN).fill(0);
			var resourceRef = emnapiCtx.createReference(envObject, resource, 1, 1);
			var resource_ = resourceRef.id;
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			HEAP_DATA_VIEW.setUint32(tsfn, resource_, true);
			if (!emnapiTSFN.initQueue(tsfn)) {
				_free(tsfn);
				resourceRef.dispose();
				return envObject.setLastError(9);
			}
			_emnapi_node_emit_async_init(resource, resource_name, -1, tsfn + emnapiTSFN.offset.async_id);
			HEAP_DATA_VIEW.setUint32(tsfn + emnapiTSFN.offset.thread_count, initial_thread_count, true);
			HEAP_DATA_VIEW.setUint32(tsfn + emnapiTSFN.offset.context, context$1, true);
			HEAP_DATA_VIEW.setUint32(tsfn + emnapiTSFN.offset.max_queue_size, max_queue_size, true);
			HEAP_DATA_VIEW.setUint32(tsfn + emnapiTSFN.offset.ref, ref, true);
			HEAP_DATA_VIEW.setUint32(tsfn + emnapiTSFN.offset.env, env, true);
			HEAP_DATA_VIEW.setUint32(tsfn + emnapiTSFN.offset.finalize_data, thread_finalize_data, true);
			HEAP_DATA_VIEW.setUint32(tsfn + emnapiTSFN.offset.finalize_cb, thread_finalize_cb, true);
			HEAP_DATA_VIEW.setUint32(tsfn + emnapiTSFN.offset.call_js_cb, call_js_cb, true);
			emnapiCtx.addCleanupHook(envObject, emnapiTSFN.cleanup, tsfn);
			envObject.ref();
			emnapiCtx.increaseWaitingRequestCounter();
			HEAP_DATA_VIEW.setInt32(tsfn + emnapiTSFN.offset.async_ref, 1, true);
			HEAP_DATA_VIEW.setUint32(result, tsfn, true);
			return envObject.clearLastError();
		}
		/** @__sig ipp */
		function napi_get_threadsafe_function_context(func, result) {
			if (!func || !result) {
				abort();
				return 1;
			}
			var context$1 = emnapiTSFN.getContext(func);
			new DataView(wasmMemory.buffer).setUint32(result, context$1, true);
			return 0;
		}
		/** @__sig ippi */
		function napi_call_threadsafe_function(func, data, mode) {
			if (!func) {
				abort();
				return 1;
			}
			return emnapiTSFN.push(func, data, mode);
		}
		/** @__sig ip */
		function napi_acquire_threadsafe_function(func) {
			if (!func) {
				abort();
				return 1;
			}
			return emnapiTSFN.getMutex(func).execute(function() {
				if (emnapiTSFN.getIsClosing(func)) return 16;
				emnapiTSFN.addThreadCount(func);
				return 0;
			});
		}
		/** @__sig ipi */
		function napi_release_threadsafe_function(func, mode) {
			if (!func) {
				abort();
				return 1;
			}
			var mutex = emnapiTSFN.getMutex(func);
			var cond = emnapiTSFN.getCond(func);
			return mutex.execute(function() {
				if (emnapiTSFN.getThreadCount(func) === 0) return 1;
				emnapiTSFN.subThreadCount(func);
				if (emnapiTSFN.getThreadCount(func) === 0 || mode === 1) {
					if (!emnapiTSFN.getIsClosing(func)) {
						var isClosingValue = mode === 1 ? 1 : 0;
						emnapiTSFN.setIsClosing(func, isClosingValue);
						if (isClosingValue && emnapiTSFN.getMaxQueueSize(func) > 0) cond.signal();
						emnapiTSFN.send(func);
					}
				}
				return 0;
			});
		}
		/** @__sig ipp */
		function napi_unref_threadsafe_function(env, func) {
			if (!func) {
				abort();
				return 1;
			}
			var asyncRefOffset = func + emnapiTSFN.offset.async_ref >> 2;
			var arr = new Int32Array(wasmMemory.buffer);
			if (Atomics.load(arr, asyncRefOffset)) {
				Atomics.store(arr, asyncRefOffset, 0);
				emnapiCtx.decreaseWaitingRequestCounter();
			}
			return 0;
		}
		/** @__sig ipp */
		function napi_ref_threadsafe_function(env, func) {
			if (!func) {
				abort();
				return 1;
			}
			var asyncRefOffset = func + emnapiTSFN.offset.async_ref >> 2;
			var arr = new Int32Array(wasmMemory.buffer);
			if (!Atomics.load(arr, asyncRefOffset)) {
				Atomics.store(arr, asyncRefOffset, 1);
				emnapiCtx.increaseWaitingRequestCounter();
			}
			return 0;
		}
		var emnapiAWMT = {
			unusedWorkers: [],
			runningWorkers: [],
			workQueue: [],
			workerReady: null,
			offset: {
				resource: 0,
				async_id: 8,
				trigger_async_id: 16,
				env: 24,
				data: 28,
				execute: 32,
				complete: 36,
				end: 40
			},
			init: function() {
				emnapiAWMT.unusedWorkers = [];
				emnapiAWMT.runningWorkers = [];
				emnapiAWMT.workQueue = [];
				emnapiAWMT.workerReady = null;
			},
			addListener: function(worker) {
				if (!worker) return false;
				if (worker._emnapiAWMTListener) return true;
				var handler = function(e) {
					var __emnapi__ = (ENVIRONMENT_IS_NODE$1 ? e : e.data).__emnapi__;
					if (__emnapi__) {
						var type = __emnapi__.type;
						var payload = __emnapi__.payload;
						if (type === "async-work-complete") {
							emnapiCtx.decreaseWaitingRequestCounter();
							emnapiAWMT.runningWorkers.splice(emnapiAWMT.runningWorkers.indexOf(worker), 1);
							emnapiAWMT.unusedWorkers.push(worker);
							emnapiAWMT.checkIdleWorker();
							emnapiAWMT.callComplete(payload.work, 0);
						} else if (type === "async-work-queue") emnapiAWMT.scheduleWork(payload.work);
						else if (type === "async-work-cancel") emnapiAWMT.cancelWork(payload.work);
					}
				};
				var dispose = function() {
					if (ENVIRONMENT_IS_NODE$1) worker.off("message", handler);
					else worker.removeEventListener("message", handler, false);
					delete worker._emnapiAWMTListener;
				};
				worker._emnapiAWMTListener = {
					handler,
					dispose
				};
				if (ENVIRONMENT_IS_NODE$1) worker.on("message", handler);
				else worker.addEventListener("message", handler, false);
				return true;
			},
			initWorkers: function(n) {
				if (ENVIRONMENT_IS_PTHREAD) return emnapiAWMT.workerReady || (emnapiAWMT.workerReady = Promise.resolve());
				if (emnapiAWMT.workerReady) return emnapiAWMT.workerReady;
				if (typeof onCreateWorker !== "function") throw new TypeError("`options.onCreateWorker` is not a function");
				var promises = [];
				var args = [];
				if (!("emnapi_async_worker_create" in wasmInstance.exports)) throw new TypeError("`emnapi_async_worker_create` is not exported, please try to add `--export=emnapi_async_worker_create` to linker flags");
				for (var i = 0; i < n; ++i) args.push(wasmInstance.exports.emnapi_async_worker_create());
				try {
					var _loop_1 = function(i$1) {
						var worker = onCreateWorker({
							type: "async-work",
							name: "emnapi-async-worker"
						});
						var p = PThread.loadWasmModuleToWorker(worker);
						emnapiAWMT.addListener(worker);
						promises.push(p.then(function() {
							if (typeof worker.unref === "function") worker.unref();
						}));
						emnapiAWMT.unusedWorkers.push(worker);
						var arg$1 = args[i$1];
						worker.threadBlockBase = arg$1;
						worker.postMessage({ __emnapi__: {
							type: "async-worker-init",
							payload: { arg: arg$1 }
						} });
					};
					for (var i = 0; i < n; ++i) _loop_1(i);
				} catch (err$1) {
					for (var i = 0; i < n; ++i) {
						var arg = args[i];
						_free(arg);
					}
					throw err$1;
				}
				emnapiAWMT.workerReady = Promise.all(promises);
				return emnapiAWMT.workerReady;
			},
			checkIdleWorker: function() {
				if (emnapiAWMT.unusedWorkers.length > 0 && emnapiAWMT.workQueue.length > 0) {
					var worker = emnapiAWMT.unusedWorkers.shift();
					var work = emnapiAWMT.workQueue.shift();
					emnapiAWMT.runningWorkers.push(worker);
					worker.postMessage({ __emnapi__: {
						type: "async-work-execute",
						payload: { work }
					} });
				}
			},
			getResource: function(work) {
				return emnapiTSFN.loadSizeTypeValue(work + emnapiAWMT.offset.resource, false);
			},
			getExecute: function(work) {
				return emnapiTSFN.loadSizeTypeValue(work + emnapiAWMT.offset.execute, false);
			},
			getComplete: function(work) {
				return emnapiTSFN.loadSizeTypeValue(work + emnapiAWMT.offset.complete, false);
			},
			getEnv: function(work) {
				return emnapiTSFN.loadSizeTypeValue(work + emnapiAWMT.offset.env, false);
			},
			getData: function(work) {
				return emnapiTSFN.loadSizeTypeValue(work + emnapiAWMT.offset.data, false);
			},
			scheduleWork: function(work) {
				var _a;
				if (ENVIRONMENT_IS_PTHREAD) {
					var postMessage_1 = napiModule.postMessage;
					postMessage_1({ __emnapi__: {
						type: "async-work-queue",
						payload: { work }
					} });
					return;
				}
				emnapiCtx.increaseWaitingRequestCounter();
				emnapiAWMT.workQueue.push(work);
				if ((_a = emnapiAWMT.workerReady) === null || _a === void 0 ? void 0 : _a.ready) emnapiAWMT.checkIdleWorker();
				else {
					var fail = function(err$1) {
						emnapiCtx.decreaseWaitingRequestCounter();
						throw err$1;
					};
					try {
						emnapiAWMT.initWorkers(_emnapi_async_work_pool_size()).then(function() {
							emnapiAWMT.workerReady.ready = true;
							emnapiAWMT.checkIdleWorker();
						}, fail);
					} catch (err$1) {
						fail(err$1);
					}
				}
			},
			cancelWork: function(work) {
				if (ENVIRONMENT_IS_PTHREAD) {
					var postMessage_2 = napiModule.postMessage;
					postMessage_2({ __emnapi__: {
						type: "async-work-cancel",
						payload: { work }
					} });
					return 0;
				}
				var index = emnapiAWMT.workQueue.indexOf(work);
				if (index !== -1) {
					emnapiAWMT.workQueue.splice(index, 1);
					emnapiCtx.feature.setImmediate(function() {
						emnapiCtx.decreaseWaitingRequestCounter();
						emnapiAWMT.checkIdleWorker();
						emnapiAWMT.callComplete(work, 11);
					});
					return 0;
				}
				return 9;
			},
			callComplete: function(work, status) {
				var complete = emnapiAWMT.getComplete(work);
				var env = emnapiAWMT.getEnv(work);
				var data = emnapiAWMT.getData(work);
				var envObject = emnapiCtx.envStore.get(env);
				var scope = emnapiCtx.openScope(envObject);
				var callback = function() {
					if (!complete) return;
					envObject.callbackIntoModule(true, function() {
						wasmTable.get(complete)(env, status, data);
					});
				};
				try {
					if (emnapiNodeBinding) {
						var resource = emnapiAWMT.getResource(work);
						var resource_value = emnapiCtx.refStore.get(resource).get();
						var resourceObject = emnapiCtx.handleStore.get(resource_value).value;
						var view = new DataView(wasmMemory.buffer);
						var asyncId = view.getFloat64(work + emnapiAWMT.offset.async_id, true);
						var triggerAsyncId = view.getFloat64(work + emnapiAWMT.offset.trigger_async_id, true);
						emnapiNodeBinding.node.makeCallback(resourceObject, callback, [], {
							asyncId,
							triggerAsyncId
						});
					} else callback();
				} finally {
					emnapiCtx.closeScope(envObject, scope);
				}
			}
		};
		/** @__sig ippppppp */
		var napi_create_async_work = singleThreadAsyncWork ? function(env, resource, resource_name, execute, complete, data, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!execute) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var resourceObject;
			if (resource) resourceObject = Object(emnapiCtx.handleStore.get(resource).value);
			else resourceObject = {};
			if (!resource_name) return envObject.setLastError(1);
			var resourceName = String(emnapiCtx.handleStore.get(resource_name).value);
			var id = emnapiAWST.create(env, resourceObject, resourceName, execute, complete, data);
			new DataView(wasmMemory.buffer).setUint32(result, id, true);
			return envObject.clearLastError();
		} : function(env, resource, resource_name, execute, complete, data, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!execute) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var resourceObject;
			if (resource) resourceObject = Object(emnapiCtx.handleStore.get(resource).value);
			else resourceObject = {};
			if (!resource_name) return envObject.setLastError(1);
			var sizeofAW = emnapiAWMT.offset.end;
			var aw = _malloc(sizeofAW);
			if (!aw) return envObject.setLastError(9);
			new Uint8Array(wasmMemory.buffer).subarray(aw, aw + sizeofAW).fill(0);
			var s = envObject.ensureHandleId(resourceObject);
			var resource_ = emnapiCtx.createReference(envObject, s, 1, 1).id;
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			HEAP_DATA_VIEW.setUint32(aw, resource_, true);
			_emnapi_node_emit_async_init(s, resource_name, -1, aw + emnapiAWMT.offset.async_id);
			HEAP_DATA_VIEW.setUint32(aw + emnapiAWMT.offset.env, env, true);
			HEAP_DATA_VIEW.setUint32(aw + emnapiAWMT.offset.execute, execute, true);
			HEAP_DATA_VIEW.setUint32(aw + emnapiAWMT.offset.complete, complete, true);
			HEAP_DATA_VIEW.setUint32(aw + emnapiAWMT.offset.data, data, true);
			HEAP_DATA_VIEW.setUint32(result, aw, true);
			return envObject.clearLastError();
		};
		/** @__sig ipp */
		var napi_delete_async_work = singleThreadAsyncWork ? function(env, work) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!work) return envObject.setLastError(1);
			emnapiAWST.remove(work);
			return envObject.clearLastError();
		} : function(env, work) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!work) return envObject.setLastError(1);
			var resource = emnapiAWMT.getResource(work);
			emnapiCtx.refStore.get(resource).dispose();
			if (emnapiNodeBinding) {
				var view = new DataView(wasmMemory.buffer);
				var asyncId = view.getFloat64(work + emnapiAWMT.offset.async_id, true);
				var triggerAsyncId = view.getFloat64(work + emnapiAWMT.offset.trigger_async_id, true);
				_emnapi_node_emit_async_destroy(asyncId, triggerAsyncId);
			}
			_free(work);
			return envObject.clearLastError();
		};
		/** @__sig ipp */
		var napi_queue_async_work = singleThreadAsyncWork ? function(env, work) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!work) return envObject.setLastError(1);
			emnapiAWST.queue(work);
			return envObject.clearLastError();
		} : function(env, work) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!work) return envObject.setLastError(1);
			emnapiAWMT.scheduleWork(work);
			return envObject.clearLastError();
		};
		/** @__sig ipp */
		var napi_cancel_async_work = singleThreadAsyncWork ? function(env, work) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!work) return envObject.setLastError(1);
			var status = emnapiAWST.cancel(work);
			if (status === 0) return envObject.clearLastError();
			return envObject.setLastError(status);
		} : function(env, work) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!work) return envObject.setLastError(1);
			var status = emnapiAWMT.cancelWork(work);
			if (status === 0) return envObject.clearLastError();
			return envObject.setLastError(status);
		};
		function initWorker(startArg) {
			if (napiModule.childThread) {
				if (typeof wasmInstance.exports.emnapi_async_worker_init !== "function") throw new TypeError("`emnapi_async_worker_init` is not exported, please try to add `--export=emnapi_async_worker_init` to linker flags");
				wasmInstance.exports.emnapi_async_worker_init(startArg);
			} else throw new Error("startThread is only available in child threads");
		}
		function executeAsyncWork(work) {
			if (!ENVIRONMENT_IS_PTHREAD) return;
			var execute = emnapiAWMT.getExecute(work);
			var env = emnapiAWMT.getEnv(work);
			var data = emnapiAWMT.getData(work);
			wasmTable.get(execute)(env, data);
			var postMessage$1 = napiModule.postMessage;
			postMessage$1({ __emnapi__: {
				type: "async-work-complete",
				payload: { work }
			} });
		}
		napiModule.initWorker = initWorker;
		napiModule.executeAsyncWork = executeAsyncWork;
		var asyncWorkMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_cancel_async_work,
			napi_create_async_work,
			napi_delete_async_work,
			napi_queue_async_work
		});
		/**
		* @__deps malloc
		* @__deps free
		* @__postset
		* ```
		* emnapiExternalMemory.init();
		* ```
		*/
		var emnapiExternalMemory = {
			registry: typeof FinalizationRegistry === "function" ? new FinalizationRegistry(function(_pointer) {
				_free(_pointer);
			}) : void 0,
			table: /* @__PURE__ */ new WeakMap(),
			wasmMemoryViewTable: /* @__PURE__ */ new WeakMap(),
			init: function() {
				emnapiExternalMemory.registry = typeof FinalizationRegistry === "function" ? new FinalizationRegistry(function(_pointer) {
					_free(_pointer);
				}) : void 0;
				emnapiExternalMemory.table = /* @__PURE__ */ new WeakMap();
				emnapiExternalMemory.wasmMemoryViewTable = /* @__PURE__ */ new WeakMap();
			},
			isDetachedArrayBuffer: function(arrayBuffer) {
				if (arrayBuffer.byteLength === 0) try {
					new Uint8Array(arrayBuffer);
				} catch (_) {
					return true;
				}
				return false;
			},
			getArrayBufferPointer: function(arrayBuffer, shouldCopy) {
				var _a;
				var info = {
					address: 0,
					ownership: 0,
					runtimeAllocated: 0
				};
				if (arrayBuffer === wasmMemory.buffer) return info;
				var isDetached = emnapiExternalMemory.isDetachedArrayBuffer(arrayBuffer);
				if (emnapiExternalMemory.table.has(arrayBuffer)) {
					var cachedInfo = emnapiExternalMemory.table.get(arrayBuffer);
					if (isDetached) {
						cachedInfo.address = 0;
						return cachedInfo;
					}
					if (shouldCopy && cachedInfo.ownership === 0 && cachedInfo.runtimeAllocated === 1) new Uint8Array(wasmMemory.buffer).set(new Uint8Array(arrayBuffer), cachedInfo.address);
					return cachedInfo;
				}
				if (isDetached || arrayBuffer.byteLength === 0) return info;
				if (!shouldCopy) return info;
				var pointer = _malloc(arrayBuffer.byteLength);
				if (!pointer) throw new Error("Out of memory");
				new Uint8Array(wasmMemory.buffer).set(new Uint8Array(arrayBuffer), pointer);
				info.address = pointer;
				info.ownership = emnapiExternalMemory.registry ? 0 : 1;
				info.runtimeAllocated = 1;
				emnapiExternalMemory.table.set(arrayBuffer, info);
				(_a = emnapiExternalMemory.registry) === null || _a === void 0 || _a.register(arrayBuffer, pointer);
				return info;
			},
			getOrUpdateMemoryView: function(view) {
				if (view.buffer === wasmMemory.buffer) {
					if (!emnapiExternalMemory.wasmMemoryViewTable.has(view)) emnapiExternalMemory.wasmMemoryViewTable.set(view, {
						Ctor: view.constructor,
						address: view.byteOffset,
						length: view instanceof DataView ? view.byteLength : view.length,
						ownership: 1,
						runtimeAllocated: 0
					});
					return view;
				}
				if ((emnapiExternalMemory.isDetachedArrayBuffer(view.buffer) || typeof SharedArrayBuffer === "function" && view.buffer instanceof SharedArrayBuffer) && emnapiExternalMemory.wasmMemoryViewTable.has(view)) {
					var info = emnapiExternalMemory.wasmMemoryViewTable.get(view);
					var Ctor = info.Ctor;
					var newView = void 0;
					var Buffer_1 = emnapiCtx.feature.Buffer;
					if (typeof Buffer_1 === "function" && Ctor === Buffer_1) newView = Buffer_1.from(wasmMemory.buffer, info.address, info.length);
					else newView = new Ctor(wasmMemory.buffer, info.address, info.length);
					emnapiExternalMemory.wasmMemoryViewTable.set(newView, info);
					return newView;
				}
				return view;
			},
			getViewPointer: function(view, shouldCopy) {
				view = emnapiExternalMemory.getOrUpdateMemoryView(view);
				if (view.buffer === wasmMemory.buffer) {
					if (emnapiExternalMemory.wasmMemoryViewTable.has(view)) {
						var _a = emnapiExternalMemory.wasmMemoryViewTable.get(view), address_1 = _a.address, ownership_1 = _a.ownership, runtimeAllocated_1 = _a.runtimeAllocated;
						return {
							address: address_1,
							ownership: ownership_1,
							runtimeAllocated: runtimeAllocated_1,
							view
						};
					}
					return {
						address: view.byteOffset,
						ownership: 1,
						runtimeAllocated: 0,
						view
					};
				}
				var _b = emnapiExternalMemory.getArrayBufferPointer(view.buffer, shouldCopy), address = _b.address, ownership = _b.ownership, runtimeAllocated = _b.runtimeAllocated;
				return {
					address: address === 0 ? 0 : address + view.byteOffset,
					ownership,
					runtimeAllocated,
					view
				};
			}
		};
		/**
		* @__postset
		* ```
		* emnapiString.init();
		* ```
		*/
		var emnapiString = {
			utf8Decoder: void 0,
			utf16Decoder: void 0,
			init: function() {
				emnapiString.utf8Decoder = typeof TextDecoder === "function" ? new TextDecoder() : { decode: function(bytes) {
					var inputIndex = 0;
					var pendingSize = Math.min(4096, bytes.length + 1);
					var pending = new Uint16Array(pendingSize);
					var chunks = [];
					var pendingIndex = 0;
					for (;;) {
						var more = inputIndex < bytes.length;
						if (!more || pendingIndex >= pendingSize - 1) {
							var arraylike = pending.subarray(0, pendingIndex);
							chunks.push(String.fromCharCode.apply(null, arraylike));
							if (!more) return chunks.join("");
							bytes = bytes.subarray(inputIndex);
							inputIndex = 0;
							pendingIndex = 0;
						}
						var byte1 = bytes[inputIndex++];
						if ((byte1 & 128) === 0) pending[pendingIndex++] = byte1;
						else if ((byte1 & 224) === 192) {
							var byte2 = bytes[inputIndex++] & 63;
							pending[pendingIndex++] = (byte1 & 31) << 6 | byte2;
						} else if ((byte1 & 240) === 224) {
							var byte2 = bytes[inputIndex++] & 63;
							var byte3 = bytes[inputIndex++] & 63;
							pending[pendingIndex++] = (byte1 & 31) << 12 | byte2 << 6 | byte3;
						} else if ((byte1 & 248) === 240) {
							var byte2 = bytes[inputIndex++] & 63;
							var byte3 = bytes[inputIndex++] & 63;
							var byte4 = bytes[inputIndex++] & 63;
							var codepoint = (byte1 & 7) << 18 | byte2 << 12 | byte3 << 6 | byte4;
							if (codepoint > 65535) {
								codepoint -= 65536;
								pending[pendingIndex++] = codepoint >>> 10 & 1023 | 55296;
								codepoint = 56320 | codepoint & 1023;
							}
							pending[pendingIndex++] = codepoint;
						}
					}
				} };
				emnapiString.utf16Decoder = typeof TextDecoder === "function" ? new TextDecoder("utf-16le") : { decode: function(input) {
					var bytes = new Uint16Array(input.buffer, input.byteOffset, input.byteLength / 2);
					if (bytes.length <= 4096) return String.fromCharCode.apply(null, bytes);
					var chunks = [];
					var i = 0;
					var len = 0;
					for (; i < bytes.length; i += len) {
						len = Math.min(4096, bytes.length - i);
						chunks.push(String.fromCharCode.apply(null, bytes.subarray(i, i + len)));
					}
					return chunks.join("");
				} };
			},
			lengthBytesUTF8: function(str) {
				var c;
				var len = 0;
				for (var i = 0; i < str.length; ++i) {
					c = str.charCodeAt(i);
					if (c <= 127) len++;
					else if (c <= 2047) len += 2;
					else if (c >= 55296 && c <= 57343) {
						len += 4;
						++i;
					} else len += 3;
				}
				return len;
			},
			UTF8ToString: function(ptr, length) {
				if (!ptr || !length) return "";
				ptr >>>= 0;
				var HEAPU8 = new Uint8Array(wasmMemory.buffer);
				var end = ptr;
				if (length === -1) for (; HEAPU8[end];) ++end;
				else end = ptr + (length >>> 0);
				length = end - ptr;
				if (length <= 16) {
					var idx = ptr;
					var str = "";
					while (idx < end) {
						var u0 = HEAPU8[idx++];
						if (!(u0 & 128)) {
							str += String.fromCharCode(u0);
							continue;
						}
						var u1 = HEAPU8[idx++] & 63;
						if ((u0 & 224) === 192) {
							str += String.fromCharCode((u0 & 31) << 6 | u1);
							continue;
						}
						var u2 = HEAPU8[idx++] & 63;
						if ((u0 & 240) === 224) u0 = (u0 & 15) << 12 | u1 << 6 | u2;
						else u0 = (u0 & 7) << 18 | u1 << 12 | u2 << 6 | HEAPU8[idx++] & 63;
						if (u0 < 65536) str += String.fromCharCode(u0);
						else {
							var ch = u0 - 65536;
							str += String.fromCharCode(55296 | ch >> 10, 56320 | ch & 1023);
						}
					}
					return str;
				}
				return emnapiString.utf8Decoder.decode(typeof SharedArrayBuffer === "function" && HEAPU8.buffer instanceof SharedArrayBuffer || Object.prototype.toString.call(HEAPU8.buffer) === "[object SharedArrayBuffer]" ? HEAPU8.slice(ptr, end) : HEAPU8.subarray(ptr, end));
			},
			stringToUTF8: function(str, outPtr, maxBytesToWrite) {
				var HEAPU8 = new Uint8Array(wasmMemory.buffer);
				var outIdx = outPtr;
				outIdx >>>= 0;
				if (!(maxBytesToWrite > 0)) return 0;
				var startIdx = outIdx;
				var endIdx = outIdx + maxBytesToWrite - 1;
				for (var i = 0; i < str.length; ++i) {
					var u = str.charCodeAt(i);
					if (u >= 55296 && u <= 57343) {
						var u1 = str.charCodeAt(++i);
						u = 65536 + ((u & 1023) << 10) | u1 & 1023;
					}
					if (u <= 127) {
						if (outIdx >= endIdx) break;
						HEAPU8[outIdx++] = u;
					} else if (u <= 2047) {
						if (outIdx + 1 >= endIdx) break;
						HEAPU8[outIdx++] = 192 | u >> 6;
						HEAPU8[outIdx++] = 128 | u & 63;
					} else if (u <= 65535) {
						if (outIdx + 2 >= endIdx) break;
						HEAPU8[outIdx++] = 224 | u >> 12;
						HEAPU8[outIdx++] = 128 | u >> 6 & 63;
						HEAPU8[outIdx++] = 128 | u & 63;
					} else {
						if (outIdx + 3 >= endIdx) break;
						HEAPU8[outIdx++] = 240 | u >> 18;
						HEAPU8[outIdx++] = 128 | u >> 12 & 63;
						HEAPU8[outIdx++] = 128 | u >> 6 & 63;
						HEAPU8[outIdx++] = 128 | u & 63;
					}
				}
				HEAPU8[outIdx] = 0;
				return outIdx - startIdx;
			},
			UTF16ToString: function(ptr, length) {
				if (!ptr || !length) return "";
				ptr >>>= 0;
				var end = ptr;
				if (length === -1) {
					var idx = end >> 1;
					var HEAPU16 = new Uint16Array(wasmMemory.buffer);
					while (HEAPU16[idx]) ++idx;
					end = idx << 1;
				} else end = ptr + (length >>> 0) * 2;
				length = end - ptr;
				if (length <= 32) return String.fromCharCode.apply(null, new Uint16Array(wasmMemory.buffer, ptr, length / 2));
				var HEAPU8 = new Uint8Array(wasmMemory.buffer);
				return emnapiString.utf16Decoder.decode(typeof SharedArrayBuffer === "function" && HEAPU8.buffer instanceof SharedArrayBuffer || Object.prototype.toString.call(HEAPU8.buffer) === "[object SharedArrayBuffer]" ? HEAPU8.slice(ptr, end) : HEAPU8.subarray(ptr, end));
			},
			stringToUTF16: function(str, outPtr, maxBytesToWrite) {
				if (maxBytesToWrite === void 0) maxBytesToWrite = 2147483647;
				if (maxBytesToWrite < 2) return 0;
				maxBytesToWrite -= 2;
				var startPtr = outPtr;
				var numCharsToWrite = maxBytesToWrite < str.length * 2 ? maxBytesToWrite / 2 : str.length;
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				for (var i = 0; i < numCharsToWrite; ++i) {
					var codeUnit = str.charCodeAt(i);
					HEAP_DATA_VIEW.setInt16(outPtr, codeUnit, true);
					outPtr += 2;
				}
				HEAP_DATA_VIEW.setInt16(outPtr, 0, true);
				return outPtr - startPtr;
			},
			newString: function(env, str, length, result, stringMaker) {
				if (!env) return 1;
				var envObject = emnapiCtx.envStore.get(env);
				envObject.checkGCAccess();
				var autoLength = length === -1;
				var sizelength = length >>> 0;
				if (length !== 0) {
					if (!str) return envObject.setLastError(1);
				}
				if (!result) return envObject.setLastError(1);
				if (!(autoLength || sizelength <= 2147483647)) return envObject.setLastError(1);
				var strValue = stringMaker(str, autoLength, sizelength);
				var value = emnapiCtx.addToCurrentScope(strValue).id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.clearLastError();
			},
			newExternalString: function(env, str, length, finalize_callback, finalize_hint, result, copied, createApi, stringMaker) {
				if (!env) return 1;
				var envObject = emnapiCtx.envStore.get(env);
				envObject.checkGCAccess();
				var autoLength = length === -1;
				var sizelength = length >>> 0;
				if (length !== 0) {
					if (!str) return envObject.setLastError(1);
				}
				if (!result) return envObject.setLastError(1);
				if (!(autoLength || sizelength <= 2147483647)) return envObject.setLastError(1);
				var status = createApi(env, str, length, result);
				if (status === 0) {
					if (copied) new DataView(wasmMemory.buffer).setInt8(copied, 1, true);
					if (finalize_callback) envObject.callFinalizer(finalize_callback, str, finalize_hint);
				}
				return status;
			}
		};
		/**
		* @__sig ippp
		*/
		function napi_get_array_length(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!value) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				var handle = emnapiCtx.handleStore.get(value);
				if (!handle.isArray()) return envObject.setLastError(8);
				var v = handle.value.length >>> 0;
				new DataView(wasmMemory.buffer).setUint32(result, v, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipppp
		*/
		function napi_get_arraybuffer_info(env, arraybuffer, data, byte_length) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!arraybuffer) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(arraybuffer);
			if (!handle.isArrayBuffer()) return envObject.setLastError(1);
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (data) {
				var p = emnapiExternalMemory.getArrayBufferPointer(handle.value, true).address;
				HEAP_DATA_VIEW.setUint32(data, p, true);
			}
			if (byte_length) HEAP_DATA_VIEW.setUint32(byte_length, handle.value.byteLength, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippp
		*/
		function napi_get_prototype(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!value) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				var handle = emnapiCtx.handleStore.get(value);
				if (handle.value == null) throw new TypeError("Cannot convert undefined or null to object");
				var v = void 0;
				try {
					v = handle.isObject() || handle.isFunction() ? handle.value : Object(handle.value);
				} catch (_) {
					return envObject.setLastError(2);
				}
				var p = envObject.ensureHandleId(Object.getPrototypeOf(v));
				new DataView(wasmMemory.buffer).setUint32(result, p, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ippppppp
		*/
		function napi_get_typedarray_info(env, typedarray, type, length, data, arraybuffer, byte_offset) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!typedarray) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(typedarray);
			if (!handle.isTypedArray()) return envObject.setLastError(1);
			var v = handle.value;
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (type) {
				var t = void 0;
				if (v instanceof Int8Array) t = 0;
				else if (v instanceof Uint8Array) t = 1;
				else if (v instanceof Uint8ClampedArray) t = 2;
				else if (v instanceof Int16Array) t = 3;
				else if (v instanceof Uint16Array) t = 4;
				else if (v instanceof Int32Array) t = 5;
				else if (v instanceof Uint32Array) t = 6;
				else if (v instanceof Float32Array) t = 7;
				else if (v instanceof Float64Array) t = 8;
				else if (v instanceof BigInt64Array) t = 9;
				else if (v instanceof BigUint64Array) t = 10;
				else return envObject.setLastError(9);
				HEAP_DATA_VIEW.setInt32(type, t, true);
			}
			if (length) HEAP_DATA_VIEW.setUint32(length, v.length, true);
			var buffer;
			if (data || arraybuffer) {
				buffer = v.buffer;
				if (data) {
					var p = emnapiExternalMemory.getViewPointer(v, true).address;
					HEAP_DATA_VIEW.setUint32(data, p, true);
				}
				if (arraybuffer) {
					var ab = envObject.ensureHandleId(buffer);
					HEAP_DATA_VIEW.setUint32(arraybuffer, ab, true);
				}
			}
			if (byte_offset) HEAP_DATA_VIEW.setUint32(byte_offset, v.byteOffset, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ipppp
		*/
		function napi_get_buffer_info(env, buffer, data, length) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!buffer) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(buffer);
			if (!handle.isBuffer(emnapiCtx.feature.Buffer)) return envObject.setLastError(1);
			if (handle.isDataView()) return napi_get_dataview_info(env, buffer, length, data, 0, 0);
			return napi_get_typedarray_info(env, buffer, 0, length, data, 0, 0);
		}
		/**
		* @__sig ipppppp
		*/
		function napi_get_dataview_info(env, dataview, byte_length, data, arraybuffer, byte_offset) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!dataview) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(dataview);
			if (!handle.isDataView()) return envObject.setLastError(1);
			var v = handle.value;
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (byte_length) HEAP_DATA_VIEW.setUint32(byte_length, v.byteLength, true);
			var buffer;
			if (data || arraybuffer) {
				buffer = v.buffer;
				if (data) {
					var p = emnapiExternalMemory.getViewPointer(v, true).address;
					HEAP_DATA_VIEW.setUint32(data, p, true);
				}
				if (arraybuffer) {
					var ab = envObject.ensureHandleId(buffer);
					HEAP_DATA_VIEW.setUint32(arraybuffer, ab, true);
				}
			}
			if (byte_offset) HEAP_DATA_VIEW.setUint32(byte_offset, v.byteOffset, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippp
		*/
		function napi_get_date_value(env, value, result) {
			var v;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!value) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				var handle = emnapiCtx.handleStore.get(value);
				if (!handle.isDate()) return envObject.setLastError(1);
				v = handle.value.valueOf();
				new DataView(wasmMemory.buffer).setFloat64(result, v, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ippp
		*/
		function napi_get_value_bool(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(value);
			if (typeof handle.value !== "boolean") return envObject.setLastError(7);
			var r = handle.value ? 1 : 0;
			new DataView(wasmMemory.buffer).setInt8(result, r, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippp
		*/
		function napi_get_value_double(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(value);
			if (typeof handle.value !== "number") return envObject.setLastError(6);
			var r = handle.value;
			new DataView(wasmMemory.buffer).setFloat64(result, r, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ipppp
		*/
		function napi_get_value_bigint_int64(env, value, result, lossless) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!emnapiCtx.feature.supportBigInt) return envObject.setLastError(9);
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			if (!lossless) return envObject.setLastError(1);
			var numberValue = emnapiCtx.handleStore.get(value).value;
			if (typeof numberValue !== "bigint") return envObject.setLastError(6);
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (numberValue >= BigInt(-1) * (BigInt(1) << BigInt(63)) && numberValue < BigInt(1) << BigInt(63)) HEAP_DATA_VIEW.setInt8(lossless, 1, true);
			else {
				HEAP_DATA_VIEW.setInt8(lossless, 0, true);
				numberValue = numberValue & (BigInt(1) << BigInt(64)) - BigInt(1);
				if (numberValue >= BigInt(1) << BigInt(63)) numberValue = numberValue - (BigInt(1) << BigInt(64));
			}
			var low = Number(numberValue & BigInt(4294967295));
			var high = Number(numberValue >> BigInt(32));
			HEAP_DATA_VIEW.setInt32(result, low, true);
			HEAP_DATA_VIEW.setInt32(result + 4, high, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ipppp
		*/
		function napi_get_value_bigint_uint64(env, value, result, lossless) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!emnapiCtx.feature.supportBigInt) return envObject.setLastError(9);
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			if (!lossless) return envObject.setLastError(1);
			var numberValue = emnapiCtx.handleStore.get(value).value;
			if (typeof numberValue !== "bigint") return envObject.setLastError(6);
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (numberValue >= BigInt(0) && numberValue < BigInt(1) << BigInt(64)) HEAP_DATA_VIEW.setInt8(lossless, 1, true);
			else {
				HEAP_DATA_VIEW.setInt8(lossless, 0, true);
				numberValue = numberValue & (BigInt(1) << BigInt(64)) - BigInt(1);
			}
			var low = Number(numberValue & BigInt(4294967295));
			var high = Number(numberValue >> BigInt(32));
			HEAP_DATA_VIEW.setUint32(result, low, true);
			HEAP_DATA_VIEW.setUint32(result + 4, high, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippppp
		*/
		function napi_get_value_bigint_words(env, value, sign_bit, word_count, words) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!emnapiCtx.feature.supportBigInt) return envObject.setLastError(9);
			if (!value) return envObject.setLastError(1);
			if (!word_count) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(value);
			if (!handle.isBigInt()) return envObject.setLastError(17);
			var isMinus = handle.value < BigInt(0);
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			var word_count_int = HEAP_DATA_VIEW.getUint32(word_count, true);
			var wordCount = 0;
			var bigintValue = isMinus ? handle.value * BigInt(-1) : handle.value;
			while (bigintValue !== BigInt(0)) {
				wordCount++;
				bigintValue = bigintValue >> BigInt(64);
			}
			bigintValue = isMinus ? handle.value * BigInt(-1) : handle.value;
			if (!sign_bit && !words) {
				word_count_int = wordCount;
				HEAP_DATA_VIEW.setUint32(word_count, word_count_int, true);
			} else {
				if (!sign_bit) return envObject.setLastError(1);
				if (!words) return envObject.setLastError(1);
				var wordsArr = [];
				while (bigintValue !== BigInt(0)) {
					var uint64 = bigintValue & (BigInt(1) << BigInt(64)) - BigInt(1);
					wordsArr.push(uint64);
					bigintValue = bigintValue >> BigInt(64);
				}
				var len = Math.min(word_count_int, wordsArr.length);
				for (var i = 0; i < len; i++) {
					var low = Number(wordsArr[i] & BigInt(4294967295));
					var high = Number(wordsArr[i] >> BigInt(32));
					HEAP_DATA_VIEW.setUint32(words + i * 8, low, true);
					HEAP_DATA_VIEW.setUint32(words + (i * 8 + 4), high, true);
				}
				HEAP_DATA_VIEW.setInt32(sign_bit, isMinus ? 1 : 0, true);
				HEAP_DATA_VIEW.setUint32(word_count, len, true);
			}
			return envObject.clearLastError();
		}
		/**
		* @__sig ippp
		*/
		function napi_get_value_external(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(value);
			if (!handle.isExternal()) return envObject.setLastError(1);
			var p = handle.data();
			new DataView(wasmMemory.buffer).setUint32(result, p, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippp
		*/
		function napi_get_value_int32(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(value);
			if (typeof handle.value !== "number") return envObject.setLastError(6);
			var v = new Int32Array([handle.value])[0];
			new DataView(wasmMemory.buffer).setInt32(result, v, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippp
		*/
		function napi_get_value_int64(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(value);
			if (typeof handle.value !== "number") return envObject.setLastError(6);
			var numberValue = handle.value;
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (numberValue === Number.POSITIVE_INFINITY || numberValue === Number.NEGATIVE_INFINITY || isNaN(numberValue)) {
				HEAP_DATA_VIEW.setInt32(result, 0, true);
				HEAP_DATA_VIEW.setInt32(result + 4, 0, true);
			} else if (numberValue < -0x8000000000000000) {
				HEAP_DATA_VIEW.setInt32(result, 0, true);
				HEAP_DATA_VIEW.setInt32(result + 4, 2147483648, true);
			} else if (numberValue >= 0x8000000000000000) {
				HEAP_DATA_VIEW.setUint32(result, 4294967295, true);
				HEAP_DATA_VIEW.setUint32(result + 4, 2147483647, true);
			} else $emnapiSetValueI64(result, Math.trunc(numberValue));
			return envObject.clearLastError();
		}
		/**
		* @__sig ippppp
		*/
		function napi_get_value_string_latin1(env, value, buf, buf_size, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			buf_size = buf_size >>> 0;
			var handle = emnapiCtx.handleStore.get(value);
			if (typeof handle.value !== "string") return envObject.setLastError(3);
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (!buf) {
				if (!result) return envObject.setLastError(1);
				HEAP_DATA_VIEW.setUint32(result, handle.value.length, true);
			} else if (buf_size !== 0) {
				var copied = 0;
				var v = void 0;
				for (var i = 0; i < buf_size - 1; ++i) {
					v = handle.value.charCodeAt(i) & 255;
					HEAP_DATA_VIEW.setUint8(buf + i, v, true);
					copied++;
				}
				HEAP_DATA_VIEW.setUint8(buf + copied, 0, true);
				if (result) HEAP_DATA_VIEW.setUint32(result, copied, true);
			} else if (result) HEAP_DATA_VIEW.setUint32(result, 0, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippppp
		*/
		function napi_get_value_string_utf8(env, value, buf, buf_size, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			buf_size = buf_size >>> 0;
			var handle = emnapiCtx.handleStore.get(value);
			if (typeof handle.value !== "string") return envObject.setLastError(3);
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (!buf) {
				if (!result) return envObject.setLastError(1);
				var strLength = emnapiString.lengthBytesUTF8(handle.value);
				HEAP_DATA_VIEW.setUint32(result, strLength, true);
			} else if (buf_size !== 0) {
				var copied = emnapiString.stringToUTF8(handle.value, buf, buf_size);
				if (result) HEAP_DATA_VIEW.setUint32(result, copied, true);
			} else if (result) HEAP_DATA_VIEW.setUint32(result, 0, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippppp
		*/
		function napi_get_value_string_utf16(env, value, buf, buf_size, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			buf_size = buf_size >>> 0;
			var handle = emnapiCtx.handleStore.get(value);
			if (typeof handle.value !== "string") return envObject.setLastError(3);
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (!buf) {
				if (!result) return envObject.setLastError(1);
				HEAP_DATA_VIEW.setUint32(result, handle.value.length, true);
			} else if (buf_size !== 0) {
				var copied = emnapiString.stringToUTF16(handle.value, buf, buf_size * 2);
				if (result) HEAP_DATA_VIEW.setUint32(result, copied / 2, true);
			} else if (result) HEAP_DATA_VIEW.setUint32(result, 0, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippp
		*/
		function napi_get_value_uint32(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(value);
			if (typeof handle.value !== "number") return envObject.setLastError(6);
			var v = new Uint32Array([handle.value])[0];
			new DataView(wasmMemory.buffer).setUint32(result, v, true);
			return envObject.clearLastError();
		}
		var convert2cMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_get_array_length,
			napi_get_arraybuffer_info,
			napi_get_buffer_info,
			napi_get_dataview_info,
			napi_get_date_value,
			napi_get_prototype,
			napi_get_typedarray_info,
			napi_get_value_bigint_int64,
			napi_get_value_bigint_uint64,
			napi_get_value_bigint_words,
			napi_get_value_bool,
			napi_get_value_double,
			napi_get_value_external,
			napi_get_value_int32,
			napi_get_value_int64,
			napi_get_value_string_latin1,
			napi_get_value_string_utf16,
			napi_get_value_string_utf8,
			napi_get_value_uint32
		});
		/**
		* @__sig ipip
		*/
		function napi_create_int32(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var v = emnapiCtx.addToCurrentScope(value).id;
			new DataView(wasmMemory.buffer).setUint32(result, v, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ipip
		*/
		function napi_create_uint32(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var v = emnapiCtx.addToCurrentScope(value >>> 0).id;
			new DataView(wasmMemory.buffer).setUint32(result, v, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ipjp
		*/
		function napi_create_int64(env, low, high, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			var value;
			if (!high) return envObject.setLastError(1);
			value = Number(low);
			var v1 = emnapiCtx.addToCurrentScope(value).id;
			new DataView(wasmMemory.buffer).setUint32(high, v1, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ipdp
		*/
		function napi_create_double(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var v = emnapiCtx.addToCurrentScope(value).id;
			new DataView(wasmMemory.buffer).setUint32(result, v, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ipppp
		*/
		function napi_create_string_latin1(env, str, length, result) {
			return emnapiString.newString(env, str, length, result, function(str$1, autoLength, sizeLength) {
				var latin1String = "";
				var len = 0;
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				if (autoLength) while (true) {
					var ch = HEAP_DATA_VIEW.getUint8(str$1, true);
					if (!ch) break;
					latin1String += String.fromCharCode(ch);
					str$1++;
				}
				else while (len < sizeLength) {
					var ch = HEAP_DATA_VIEW.getUint8(str$1, true);
					if (!ch) break;
					latin1String += String.fromCharCode(ch);
					len++;
					str$1++;
				}
				return latin1String;
			});
		}
		/**
		* @__sig ipppp
		*/
		function napi_create_string_utf16(env, str, length, result) {
			return emnapiString.newString(env, str, length, result, function(str$1) {
				return emnapiString.UTF16ToString(str$1, length);
			});
		}
		/**
		* @__sig ipppp
		*/
		function napi_create_string_utf8(env, str, length, result) {
			return emnapiString.newString(env, str, length, result, function(str$1) {
				return emnapiString.UTF8ToString(str$1, length);
			});
		}
		/**
		* @__sig ippppppp
		*/
		function node_api_create_external_string_latin1(env, str, length, finalize_callback, finalize_hint, result, copied) {
			return emnapiString.newExternalString(env, str, length, finalize_callback, finalize_hint, result, copied, napi_create_string_latin1, void 0);
		}
		/**
		* @__sig ippppppp
		*/
		function node_api_create_external_string_utf16(env, str, length, finalize_callback, finalize_hint, result, copied) {
			return emnapiString.newExternalString(env, str, length, finalize_callback, finalize_hint, result, copied, napi_create_string_utf16, void 0);
		}
		/**
		* @__sig ipppp
		*/
		function node_api_create_property_key_latin1(env, str, length, result) {
			return napi_create_string_latin1(env, str, length, result);
		}
		/**
		* @__sig ipppp
		*/
		function node_api_create_property_key_utf8(env, str, length, result) {
			return napi_create_string_utf8(env, str, length, result);
		}
		/**
		* @__sig ipppp
		*/
		function node_api_create_property_key_utf16(env, str, length, result) {
			return napi_create_string_utf16(env, str, length, result);
		}
		/**
		* @__sig ipjp
		*/
		function napi_create_bigint_int64(env, low, high, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!emnapiCtx.feature.supportBigInt) return envObject.setLastError(9);
			var value;
			if (!high) return envObject.setLastError(1);
			value = low;
			var v1 = emnapiCtx.addToCurrentScope(value).id;
			new DataView(wasmMemory.buffer).setUint32(high, v1, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ipjp
		*/
		function napi_create_bigint_uint64(env, low, high, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!emnapiCtx.feature.supportBigInt) return envObject.setLastError(9);
			var value;
			if (!high) return envObject.setLastError(1);
			value = low & (BigInt(1) << BigInt(64)) - BigInt(1);
			var v1 = emnapiCtx.addToCurrentScope(value).id;
			new DataView(wasmMemory.buffer).setUint32(high, v1, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ipippp
		*/
		function napi_create_bigint_words(env, sign_bit, word_count, words, result) {
			var v, i;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!emnapiCtx.feature.supportBigInt) return envObject.setLastError(9);
				if (!result) return envObject.setLastError(1);
				word_count = word_count >>> 0;
				if (word_count > 2147483647) return envObject.setLastError(1);
				if (word_count > 1024 * 1024 / 32 / 2) throw new RangeError("Maximum BigInt size exceeded");
				var value = BigInt(0);
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				for (i = 0; i < word_count; i++) {
					var low = HEAP_DATA_VIEW.getUint32(words + i * 8, true);
					var high = HEAP_DATA_VIEW.getUint32(words + (i * 8 + 4), true);
					var wordi = BigInt(low) | BigInt(high) << BigInt(32);
					value += wordi << BigInt(64 * i);
				}
				value *= BigInt(sign_bit) % BigInt(2) === BigInt(0) ? BigInt(1) : BigInt(-1);
				v = emnapiCtx.addToCurrentScope(value).id;
				HEAP_DATA_VIEW.setUint32(result, v, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		var convert2napiMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_create_bigint_int64,
			napi_create_bigint_uint64,
			napi_create_bigint_words,
			napi_create_double,
			napi_create_int32,
			napi_create_int64,
			napi_create_string_latin1,
			napi_create_string_utf16,
			napi_create_string_utf8,
			napi_create_uint32,
			node_api_create_external_string_latin1,
			node_api_create_external_string_utf16,
			node_api_create_property_key_latin1,
			node_api_create_property_key_utf16,
			node_api_create_property_key_utf8
		});
		function emnapiCreateFunction(envObject, utf8name, length, cb, data) {
			var functionName = !utf8name || !length ? "" : emnapiString.UTF8ToString(utf8name, length);
			var f;
			var napiCallback = wasmTable.get(cb);
			var callback = function(envObject$1) {
				return napiCallback(envObject$1.id, envObject$1.ctx.scopeStore.currentScope.id);
			};
			var makeFunction = function(envObject$1, callback$1) {
				return function() {
					var scope = envObject$1.ctx.openScope(envObject$1);
					var callbackInfo = scope.callbackInfo;
					callbackInfo.data = data;
					callbackInfo.args = arguments;
					callbackInfo.thiz = this;
					callbackInfo.fn = f;
					try {
						var napiValue = envObject$1.callIntoModule(callback$1);
						return !napiValue ? void 0 : envObject$1.ctx.handleStore.get(napiValue).value;
					} finally {
						callbackInfo.data = 0;
						callbackInfo.args = void 0;
						callbackInfo.thiz = void 0;
						callbackInfo.fn = void 0;
						envObject$1.ctx.closeScope(envObject$1, scope);
					}
				};
			};
			if (functionName === "") {
				f = makeFunction(envObject, callback);
				return {
					status: 0,
					f
				};
			}
			if (!/^[_$a-zA-Z][_$a-zA-Z0-9]*$/.test(functionName)) return {
				status: 1,
				f: void 0
			};
			if (emnapiCtx.feature.supportNewFunction) {
				var _ = makeFunction(envObject, callback);
				try {
					f = new Function("_", "return function " + functionName + "(){\"use strict\";return _.apply(this,arguments);};")(_);
				} catch (_err) {
					f = makeFunction(envObject, callback);
					if (emnapiCtx.feature.canSetFunctionName) Object.defineProperty(f, "name", { value: functionName });
				}
			} else {
				f = makeFunction(envObject, callback);
				if (emnapiCtx.feature.canSetFunctionName) Object.defineProperty(f, "name", { value: functionName });
			}
			return {
				status: 0,
				f
			};
		}
		function emnapiDefineProperty(envObject, obj, propertyName, method, getter, setter, value, attributes, data) {
			if (getter || setter) {
				var localGetter = void 0;
				var localSetter = void 0;
				if (getter) localGetter = emnapiCreateFunction(envObject, 0, 0, getter, data).f;
				if (setter) localSetter = emnapiCreateFunction(envObject, 0, 0, setter, data).f;
				var desc = {
					configurable: (attributes & 4) !== 0,
					enumerable: (attributes & 2) !== 0,
					get: localGetter,
					set: localSetter
				};
				Object.defineProperty(obj, propertyName, desc);
			} else if (method) {
				var localMethod = emnapiCreateFunction(envObject, 0, 0, method, data).f;
				var desc = {
					configurable: (attributes & 4) !== 0,
					enumerable: (attributes & 2) !== 0,
					writable: (attributes & 1) !== 0,
					value: localMethod
				};
				Object.defineProperty(obj, propertyName, desc);
			} else {
				var desc = {
					configurable: (attributes & 4) !== 0,
					enumerable: (attributes & 2) !== 0,
					writable: (attributes & 1) !== 0,
					value: emnapiCtx.handleStore.get(value).value
				};
				Object.defineProperty(obj, propertyName, desc);
			}
		}
		function emnapiGetHandle(js_object) {
			var handle = emnapiCtx.handleStore.get(js_object);
			if (!(handle.isObject() || handle.isFunction())) return { status: 1 };
			if (typeof emnapiExternalMemory !== "undefined" && ArrayBuffer.isView(handle.value)) {
				if (emnapiExternalMemory.wasmMemoryViewTable.has(handle.value)) handle = emnapiCtx.addToCurrentScope(emnapiExternalMemory.wasmMemoryViewTable.get(handle.value));
			}
			return {
				status: 0,
				handle
			};
		}
		function emnapiWrap(env, js_object, native_object, finalize_cb, finalize_hint, result) {
			var referenceId;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!emnapiCtx.feature.supportFinalizer) {
					if (finalize_cb) throw emnapiCtx.createNotSupportWeakRefError("napi_wrap", "Parameter \"finalize_cb\" must be 0(NULL)");
					if (result) throw emnapiCtx.createNotSupportWeakRefError("napi_wrap", "Parameter \"result\" must be 0(NULL)");
				}
				if (!js_object) return envObject.setLastError(1);
				var handleResult = emnapiGetHandle(js_object);
				if (handleResult.status !== 0) return envObject.setLastError(handleResult.status);
				var handle = handleResult.handle;
				if (envObject.getObjectBinding(handle.value).wrapped !== 0) return envObject.setLastError(1);
				var reference = void 0;
				if (result) {
					if (!finalize_cb) return envObject.setLastError(1);
					reference = emnapiCtx.createReferenceWithFinalizer(envObject, handle.id, 0, 1, finalize_cb, native_object, finalize_hint);
					referenceId = reference.id;
					new DataView(wasmMemory.buffer).setUint32(result, referenceId, true);
				} else if (finalize_cb) reference = emnapiCtx.createReferenceWithFinalizer(envObject, handle.id, 0, 0, finalize_cb, native_object, finalize_hint);
				else reference = emnapiCtx.createReferenceWithData(envObject, handle.id, 0, 0, native_object);
				envObject.getObjectBinding(handle.value).wrapped = reference.id;
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		function emnapiUnwrap(env, js_object, result, action) {
			var data;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!js_object) return envObject.setLastError(1);
				if (action === 0) {
					if (!result) return envObject.setLastError(1);
				}
				var value = emnapiCtx.handleStore.get(js_object);
				if (!(value.isObject() || value.isFunction())) return envObject.setLastError(1);
				var binding = envObject.getObjectBinding(value.value);
				var referenceId = binding.wrapped;
				var ref = emnapiCtx.refStore.get(referenceId);
				if (!ref) return envObject.setLastError(1);
				if (result) {
					data = ref.data();
					new DataView(wasmMemory.buffer).setUint32(result, data, true);
				}
				if (action === 1) {
					binding.wrapped = 0;
					if (ref.ownership() === 1) ref.resetFinalizer();
					else ref.dispose();
				}
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipppppppp
		*/
		function napi_define_class(env, utf8name, length, constructor, callback_data, property_count, properties, result) {
			var propPtr, valueHandleId, attributes;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				if (!constructor) return envObject.setLastError(1);
				property_count = property_count >>> 0;
				if (property_count > 0) {
					if (!properties) return envObject.setLastError(1);
				}
				if (length < -1 || length > 2147483647 || !utf8name) return envObject.setLastError(1);
				var fresult = emnapiCreateFunction(envObject, utf8name, length, constructor, callback_data);
				if (fresult.status !== 0) return envObject.setLastError(fresult.status);
				var F = fresult.f;
				var propertyName = void 0;
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				for (var i = 0; i < property_count; i++) {
					propPtr = properties + i * 32;
					var utf8Name = HEAP_DATA_VIEW.getUint32(propPtr, true);
					var name_1 = HEAP_DATA_VIEW.getUint32(propPtr + 4, true);
					var method = HEAP_DATA_VIEW.getUint32(propPtr + 8, true);
					var getter = HEAP_DATA_VIEW.getUint32(propPtr + 12, true);
					var setter = HEAP_DATA_VIEW.getUint32(propPtr + 16, true);
					var value = HEAP_DATA_VIEW.getUint32(propPtr + 20, true);
					attributes = HEAP_DATA_VIEW.getInt32(propPtr + 24, true);
					var data = HEAP_DATA_VIEW.getUint32(propPtr + 28, true);
					if (utf8Name) propertyName = emnapiString.UTF8ToString(utf8Name, -1);
					else {
						if (!name_1) return envObject.setLastError(4);
						propertyName = emnapiCtx.handleStore.get(name_1).value;
						if (typeof propertyName !== "string" && typeof propertyName !== "symbol") return envObject.setLastError(4);
					}
					if ((attributes & 1024) !== 0) {
						emnapiDefineProperty(envObject, F, propertyName, method, getter, setter, value, attributes, data);
						continue;
					}
					emnapiDefineProperty(envObject, F.prototype, propertyName, method, getter, setter, value, attributes, data);
				}
				valueHandleId = emnapiCtx.addToCurrentScope(F).id;
				HEAP_DATA_VIEW.setUint32(result, valueHandleId, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipppppp
		*/
		function napi_wrap(env, js_object, native_object, finalize_cb, finalize_hint, result) {
			return emnapiWrap(env, js_object, native_object, finalize_cb, finalize_hint, result);
		}
		/**
		* @__sig ippp
		*/
		function napi_unwrap(env, js_object, result) {
			return emnapiUnwrap(env, js_object, result, 0);
		}
		/**
		* @__sig ippp
		*/
		function napi_remove_wrap(env, js_object, result) {
			return emnapiUnwrap(env, js_object, result, 1);
		}
		/**
		* @__sig ippp
		*/
		function napi_type_tag_object(env, object, type_tag) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!object) return envObject.setLastError(envObject.tryCatch.hasCaught() ? 10 : 1);
				var value = emnapiCtx.handleStore.get(object);
				if (!(value.isObject() || value.isFunction())) return envObject.setLastError(envObject.tryCatch.hasCaught() ? 10 : 2);
				if (!type_tag) return envObject.setLastError(envObject.tryCatch.hasCaught() ? 10 : 1);
				var binding = envObject.getObjectBinding(value.value);
				if (binding.tag !== null) return envObject.setLastError(envObject.tryCatch.hasCaught() ? 10 : 1);
				var tag = new Uint8Array(16);
				tag.set(new Uint8Array(wasmMemory.buffer, type_tag, 16));
				binding.tag = new Uint32Array(tag.buffer);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipppp
		*/
		function napi_check_object_type_tag(env, object, type_tag, result) {
			var ret = true;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!object) return envObject.setLastError(envObject.tryCatch.hasCaught() ? 10 : 1);
				var value = emnapiCtx.handleStore.get(object);
				if (!(value.isObject() || value.isFunction())) return envObject.setLastError(envObject.tryCatch.hasCaught() ? 10 : 2);
				if (!type_tag) return envObject.setLastError(envObject.tryCatch.hasCaught() ? 10 : 1);
				if (!result) return envObject.setLastError(envObject.tryCatch.hasCaught() ? 10 : 1);
				var binding = envObject.getObjectBinding(value.value);
				if (binding.tag !== null) {
					var tag = binding.tag;
					var typeTag = new Uint32Array(wasmMemory.buffer, type_tag, 4);
					ret = tag[0] === typeTag[0] && tag[1] === typeTag[1] && tag[2] === typeTag[2] && tag[3] === typeTag[3];
				} else ret = false;
				new DataView(wasmMemory.buffer).setInt8(result, ret ? 1 : 0, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipppppp
		*/
		function napi_add_finalizer(env, js_object, finalize_data, finalize_cb, finalize_hint, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!emnapiCtx.feature.supportFinalizer) return envObject.setLastError(9);
			if (!js_object) return envObject.setLastError(1);
			if (!finalize_cb) return envObject.setLastError(1);
			var handleResult = emnapiGetHandle(js_object);
			if (handleResult.status !== 0) return envObject.setLastError(handleResult.status);
			var handle = handleResult.handle;
			var ownership = !result ? 0 : 1;
			var reference = emnapiCtx.createReferenceWithFinalizer(envObject, handle.id, 0, ownership, finalize_cb, finalize_data, finalize_hint);
			if (result) {
				var referenceId = reference.id;
				new DataView(wasmMemory.buffer).setUint32(result, referenceId, true);
			}
			return envObject.clearLastError();
		}
		/**
		* @__sig ipppp
		*/
		function node_api_post_finalizer(env, finalize_cb, finalize_data, finalize_hint) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.enqueueFinalizer(emnapiCtx.createTrackedFinalizer(envObject, finalize_cb, finalize_data, finalize_hint));
			return envObject.clearLastError();
		}
		var wrapMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_add_finalizer,
			napi_check_object_type_tag,
			napi_define_class,
			napi_remove_wrap,
			napi_type_tag_object,
			napi_unwrap,
			napi_wrap,
			node_api_post_finalizer
		});
		/**
		* @__sig ipippppp
		*/
		function emnapi_create_memory_view(env, typedarray_type, external_data, byte_length, finalize_cb, finalize_hint, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				byte_length = byte_length >>> 0;
				if (!external_data) byte_length = 0;
				if (byte_length > 2147483647) throw new RangeError("Cannot create a memory view larger than 2147483647 bytes");
				if (external_data + byte_length > wasmMemory.buffer.byteLength) throw new RangeError("Memory out of range");
				if (!emnapiCtx.feature.supportFinalizer && finalize_cb) throw emnapiCtx.createNotSupportWeakRefError("emnapi_create_memory_view", "Parameter \"finalize_cb\" must be 0(NULL)");
				var viewDescriptor = void 0;
				switch (typedarray_type) {
					case 0:
						viewDescriptor = {
							Ctor: Int8Array,
							address: external_data,
							length: byte_length,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case 1:
						viewDescriptor = {
							Ctor: Uint8Array,
							address: external_data,
							length: byte_length,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case 2:
						viewDescriptor = {
							Ctor: Uint8ClampedArray,
							address: external_data,
							length: byte_length,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case 3:
						viewDescriptor = {
							Ctor: Int16Array,
							address: external_data,
							length: byte_length >> 1,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case 4:
						viewDescriptor = {
							Ctor: Uint16Array,
							address: external_data,
							length: byte_length >> 1,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case 5:
						viewDescriptor = {
							Ctor: Int32Array,
							address: external_data,
							length: byte_length >> 2,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case 6:
						viewDescriptor = {
							Ctor: Uint32Array,
							address: external_data,
							length: byte_length >> 2,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case 7:
						viewDescriptor = {
							Ctor: Float32Array,
							address: external_data,
							length: byte_length >> 2,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case 8:
						viewDescriptor = {
							Ctor: Float64Array,
							address: external_data,
							length: byte_length >> 3,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case 9:
						viewDescriptor = {
							Ctor: BigInt64Array,
							address: external_data,
							length: byte_length >> 3,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case 10:
						viewDescriptor = {
							Ctor: BigUint64Array,
							address: external_data,
							length: byte_length >> 3,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case -1:
						viewDescriptor = {
							Ctor: DataView,
							address: external_data,
							length: byte_length,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					case -2:
						if (!emnapiCtx.feature.Buffer) throw emnapiCtx.createNotSupportBufferError("emnapi_create_memory_view", "");
						viewDescriptor = {
							Ctor: emnapiCtx.feature.Buffer,
							address: external_data,
							length: byte_length,
							ownership: 1,
							runtimeAllocated: 0
						};
						break;
					default: return envObject.setLastError(1);
				}
				var Ctor = viewDescriptor.Ctor;
				var typedArray = typedarray_type === -2 ? emnapiCtx.feature.Buffer.from(wasmMemory.buffer, viewDescriptor.address, viewDescriptor.length) : new Ctor(wasmMemory.buffer, viewDescriptor.address, viewDescriptor.length);
				var handle = emnapiCtx.addToCurrentScope(typedArray);
				emnapiExternalMemory.wasmMemoryViewTable.set(typedArray, viewDescriptor);
				if (finalize_cb) {
					var status_1 = napi_add_finalizer(env, handle.id, external_data, finalize_cb, finalize_hint, 0);
					if (status_1 === 10) {
						var err$1 = envObject.tryCatch.extractException();
						envObject.clearLastError();
						throw err$1;
					} else if (status_1 !== 0) return envObject.setLastError(status_1);
				}
				value = handle.id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$2) {
				envObject.tryCatch.setError(err$2);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig i
		*/
		function emnapi_is_support_weakref() {
			return emnapiCtx.feature.supportFinalizer ? 1 : 0;
		}
		/**
		* @__sig i
		*/
		function emnapi_is_support_bigint() {
			return emnapiCtx.feature.supportBigInt ? 1 : 0;
		}
		/**
		* @__sig i
		*/
		function emnapi_is_node_binding_available() {
			return emnapiNodeBinding ? 1 : 0;
		}
		function $emnapiSyncMemory(js_to_wasm, arrayBufferOrView, offset, len) {
			offset = offset !== null && offset !== void 0 ? offset : 0;
			offset = offset >>> 0;
			var view;
			if (arrayBufferOrView instanceof ArrayBuffer) {
				var pointer = emnapiExternalMemory.getArrayBufferPointer(arrayBufferOrView, false).address;
				if (!pointer) throw new Error("Unknown ArrayBuffer address");
				if (typeof len !== "number" || len === -1) len = arrayBufferOrView.byteLength - offset;
				len = len >>> 0;
				if (len === 0) return arrayBufferOrView;
				view = new Uint8Array(arrayBufferOrView, offset, len);
				var wasmMemoryU8 = new Uint8Array(wasmMemory.buffer);
				if (!js_to_wasm) view.set(wasmMemoryU8.subarray(pointer, pointer + len));
				else wasmMemoryU8.set(view, pointer);
				return arrayBufferOrView;
			}
			if (ArrayBuffer.isView(arrayBufferOrView)) {
				var viewPointerInfo = emnapiExternalMemory.getViewPointer(arrayBufferOrView, false);
				var latestView = viewPointerInfo.view;
				var pointer = viewPointerInfo.address;
				if (!pointer) throw new Error("Unknown ArrayBuffer address");
				if (typeof len !== "number" || len === -1) len = latestView.byteLength - offset;
				len = len >>> 0;
				if (len === 0) return latestView;
				view = new Uint8Array(latestView.buffer, latestView.byteOffset + offset, len);
				var wasmMemoryU8 = new Uint8Array(wasmMemory.buffer);
				if (!js_to_wasm) view.set(wasmMemoryU8.subarray(pointer, pointer + len));
				else wasmMemoryU8.set(view, pointer);
				return latestView;
			}
			throw new TypeError("emnapiSyncMemory expect ArrayBuffer or ArrayBufferView as first parameter");
		}
		/**
		* @__sig ipippp
		*/
		function emnapi_sync_memory(env, js_to_wasm, arraybuffer_or_view, offset, len) {
			var v;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!arraybuffer_or_view) return envObject.setLastError(1);
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				var handleId = HEAP_DATA_VIEW.getUint32(arraybuffer_or_view, true);
				var handle = envObject.ctx.handleStore.get(handleId);
				if (!handle.isArrayBuffer() && !handle.isTypedArray() && !handle.isDataView()) return envObject.setLastError(1);
				var ret = $emnapiSyncMemory(Boolean(js_to_wasm), handle.value, offset, len);
				if (handle.value !== ret) {
					v = envObject.ensureHandleId(ret);
					HEAP_DATA_VIEW.setUint32(arraybuffer_or_view, v, true);
				}
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		function $emnapiGetMemoryAddress(arrayBufferOrView) {
			var isArrayBuffer = arrayBufferOrView instanceof ArrayBuffer;
			var isDataView = arrayBufferOrView instanceof DataView;
			if (!isArrayBuffer && !(ArrayBuffer.isView(arrayBufferOrView) && !isDataView) && !isDataView) throw new TypeError("emnapiGetMemoryAddress expect ArrayBuffer or ArrayBufferView as first parameter");
			var info;
			if (isArrayBuffer) info = emnapiExternalMemory.getArrayBufferPointer(arrayBufferOrView, false);
			else info = emnapiExternalMemory.getViewPointer(arrayBufferOrView, false);
			return {
				address: info.address,
				ownership: info.ownership,
				runtimeAllocated: info.runtimeAllocated
			};
		}
		/**
		* @__sig ipppp
		*/
		function emnapi_get_memory_address(env, arraybuffer_or_view, address, ownership, runtime_allocated) {
			var p, runtimeAllocated, ownershipOut;
			var info;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!arraybuffer_or_view) return envObject.setLastError(1);
				if (!address && !ownership && !runtime_allocated) return envObject.setLastError(1);
				var handle = envObject.ctx.handleStore.get(arraybuffer_or_view);
				info = $emnapiGetMemoryAddress(handle.value);
				p = info.address;
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				if (address) HEAP_DATA_VIEW.setUint32(address, p, true);
				if (ownership) {
					ownershipOut = info.ownership;
					HEAP_DATA_VIEW.setInt32(ownership, ownershipOut, true);
				}
				if (runtime_allocated) {
					runtimeAllocated = info.runtimeAllocated;
					HEAP_DATA_VIEW.setInt8(runtime_allocated, runtimeAllocated, true);
				}
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipp
		*/
		function emnapi_get_runtime_version(env, version$3) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!version$3) return envObject.setLastError(1);
			var runtimeVersion;
			try {
				runtimeVersion = emnapiCtx.getRuntimeVersions().version;
			} catch (_) {
				return envObject.setLastError(9);
			}
			var versions = runtimeVersion.split(".").map(function(n) {
				return Number(n);
			});
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			HEAP_DATA_VIEW.setUint32(version$3, versions[0], true);
			HEAP_DATA_VIEW.setUint32(version$3 + 4, versions[1], true);
			HEAP_DATA_VIEW.setUint32(version$3 + 8, versions[2], true);
			return envObject.clearLastError();
		}
		var emnapiMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			$emnapiGetMemoryAddress,
			$emnapiSyncMemory,
			emnapi_create_memory_view,
			emnapi_get_memory_address,
			emnapi_get_runtime_version,
			emnapi_is_node_binding_available,
			emnapi_is_support_bigint,
			emnapi_is_support_weakref,
			emnapi_sync_memory
		});
		/**
		* @__sig ipp
		*/
		function napi_create_array(env, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var value = emnapiCtx.addToCurrentScope([]).id;
			new DataView(wasmMemory.buffer).setUint32(result, value, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippp
		*/
		function napi_create_array_with_length(env, length, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			length = length >>> 0;
			var value = emnapiCtx.addToCurrentScope(new Array(length)).id;
			new DataView(wasmMemory.buffer).setUint32(result, value, true);
			return envObject.clearLastError();
		}
		function emnapiCreateArrayBuffer(byte_length, data) {
			byte_length = byte_length >>> 0;
			var arrayBuffer = new ArrayBuffer(byte_length);
			if (data) {
				var p = emnapiExternalMemory.getArrayBufferPointer(arrayBuffer, true).address;
				new DataView(wasmMemory.buffer).setUint32(data, p, true);
			}
			return arrayBuffer;
		}
		/**
		* @__sig ipppp
		*/
		function napi_create_arraybuffer(env, byte_length, data, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				var arrayBuffer = emnapiCreateArrayBuffer(byte_length, data);
				value = emnapiCtx.addToCurrentScope(arrayBuffer).id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipdp
		*/
		function napi_create_date(env, time, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				value = emnapiCtx.addToCurrentScope(new Date(time)).id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ippppp
		*/
		function napi_create_external(env, data, finalize_cb, finalize_hint, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				if (!emnapiCtx.feature.supportFinalizer && finalize_cb) throw emnapiCtx.createNotSupportWeakRefError("napi_create_external", "Parameter \"finalize_cb\" must be 0(NULL)");
				var externalHandle = emnapiCtx.getCurrentScope().addExternal(data);
				if (finalize_cb) emnapiCtx.createReferenceWithFinalizer(envObject, externalHandle.id, 0, 0, finalize_cb, data, finalize_hint);
				value = externalHandle.id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.clearLastError();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipppppp
		*/
		function napi_create_external_arraybuffer(env, external_data, byte_length, finalize_cb, finalize_hint, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				byte_length = byte_length >>> 0;
				if (!external_data) byte_length = 0;
				if (external_data + byte_length > wasmMemory.buffer.byteLength) throw new RangeError("Memory out of range");
				if (!emnapiCtx.feature.supportFinalizer && finalize_cb) throw emnapiCtx.createNotSupportWeakRefError("napi_create_external_arraybuffer", "Parameter \"finalize_cb\" must be 0(NULL)");
				var arrayBuffer = new ArrayBuffer(byte_length);
				if (byte_length === 0) try {
					var MessageChannel_1 = emnapiCtx.feature.MessageChannel;
					new MessageChannel_1().port1.postMessage(arrayBuffer, [arrayBuffer]);
				} catch (_) {}
				else {
					new Uint8Array(arrayBuffer).set(new Uint8Array(wasmMemory.buffer).subarray(external_data, external_data + byte_length));
					emnapiExternalMemory.table.set(arrayBuffer, {
						address: external_data,
						ownership: 1,
						runtimeAllocated: 0
					});
				}
				var handle = emnapiCtx.addToCurrentScope(arrayBuffer);
				if (finalize_cb) {
					var status_1 = napi_add_finalizer(env, handle.id, external_data, finalize_cb, finalize_hint, 0);
					if (status_1 === 10) {
						var err$1 = envObject.tryCatch.extractException();
						envObject.clearLastError();
						throw err$1;
					} else if (status_1 !== 0) return envObject.setLastError(status_1);
				}
				value = handle.id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$2) {
				envObject.tryCatch.setError(err$2);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipp
		*/
		function napi_create_object(env, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var value = emnapiCtx.addToCurrentScope({}).id;
			new DataView(wasmMemory.buffer).setUint32(result, value, true);
			return envObject.clearLastError();
		}
		/**
		* @__sig ippp
		*/
		function napi_create_symbol(env, description, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (!description) {
				var value = emnapiCtx.addToCurrentScope(Symbol()).id;
				HEAP_DATA_VIEW.setUint32(result, value, true);
			} else {
				var desc = emnapiCtx.handleStore.get(description).value;
				if (typeof desc !== "string") return envObject.setLastError(3);
				var v = emnapiCtx.addToCurrentScope(Symbol(desc)).id;
				HEAP_DATA_VIEW.setUint32(result, v, true);
			}
			return envObject.clearLastError();
		}
		/**
		* @__sig ipipppp
		*/
		function napi_create_typedarray(env, type, length, arraybuffer, byte_offset, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!arraybuffer) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				var handle = emnapiCtx.handleStore.get(arraybuffer);
				if (!handle.isArrayBuffer()) return envObject.setLastError(1);
				var buffer = handle.value;
				var createTypedArray = function(envObject$1, Type, size_of_element, buffer$1, byte_offset$1, length$1) {
					var _a;
					byte_offset$1 = byte_offset$1 >>> 0;
					length$1 = length$1 >>> 0;
					if (size_of_element > 1) {
						if (byte_offset$1 % size_of_element !== 0) {
							var err$1 = new RangeError("start offset of ".concat((_a = Type.name) !== null && _a !== void 0 ? _a : "", " should be a multiple of ").concat(size_of_element));
							err$1.code = "ERR_NAPI_INVALID_TYPEDARRAY_ALIGNMENT";
							envObject$1.tryCatch.setError(err$1);
							return envObject$1.setLastError(9);
						}
					}
					if (length$1 * size_of_element + byte_offset$1 > buffer$1.byteLength) {
						var err$1 = /* @__PURE__ */ new RangeError("Invalid typed array length");
						err$1.code = "ERR_NAPI_INVALID_TYPEDARRAY_LENGTH";
						envObject$1.tryCatch.setError(err$1);
						return envObject$1.setLastError(9);
					}
					var out = new Type(buffer$1, byte_offset$1, length$1);
					if (buffer$1 === wasmMemory.buffer) {
						if (!emnapiExternalMemory.wasmMemoryViewTable.has(out)) emnapiExternalMemory.wasmMemoryViewTable.set(out, {
							Ctor: Type,
							address: byte_offset$1,
							length: length$1,
							ownership: 1,
							runtimeAllocated: 0
						});
					}
					value = emnapiCtx.addToCurrentScope(out).id;
					new DataView(wasmMemory.buffer).setUint32(result, value, true);
					return envObject$1.getReturnStatus();
				};
				switch (type) {
					case 0: return createTypedArray(envObject, Int8Array, 1, buffer, byte_offset, length);
					case 1: return createTypedArray(envObject, Uint8Array, 1, buffer, byte_offset, length);
					case 2: return createTypedArray(envObject, Uint8ClampedArray, 1, buffer, byte_offset, length);
					case 3: return createTypedArray(envObject, Int16Array, 2, buffer, byte_offset, length);
					case 4: return createTypedArray(envObject, Uint16Array, 2, buffer, byte_offset, length);
					case 5: return createTypedArray(envObject, Int32Array, 4, buffer, byte_offset, length);
					case 6: return createTypedArray(envObject, Uint32Array, 4, buffer, byte_offset, length);
					case 7: return createTypedArray(envObject, Float32Array, 4, buffer, byte_offset, length);
					case 8: return createTypedArray(envObject, Float64Array, 8, buffer, byte_offset, length);
					case 9: return createTypedArray(envObject, BigInt64Array, 8, buffer, byte_offset, length);
					case 10: return createTypedArray(envObject, BigUint64Array, 8, buffer, byte_offset, length);
					default: return envObject.setLastError(1);
				}
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__deps malloc
		* @__sig ippp
		*/
		function napi_create_buffer(env, size, data, result) {
			var _a;
			var value, pointer;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				var Buffer$1 = emnapiCtx.feature.Buffer;
				if (!Buffer$1) throw emnapiCtx.createNotSupportBufferError("napi_create_buffer", "");
				var buffer = void 0;
				size = size >>> 0;
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				if (!data || size === 0) {
					buffer = Buffer$1.alloc(size);
					value = emnapiCtx.addToCurrentScope(buffer).id;
					HEAP_DATA_VIEW.setUint32(result, value, true);
				} else {
					pointer = _malloc(size);
					if (!pointer) throw new Error("Out of memory");
					new Uint8Array(wasmMemory.buffer).subarray(pointer, pointer + size).fill(0);
					var buffer_1 = Buffer$1.from(wasmMemory.buffer, pointer, size);
					var viewDescriptor = {
						Ctor: Buffer$1,
						address: pointer,
						length: size,
						ownership: emnapiExternalMemory.registry ? 0 : 1,
						runtimeAllocated: 1
					};
					emnapiExternalMemory.wasmMemoryViewTable.set(buffer_1, viewDescriptor);
					(_a = emnapiExternalMemory.registry) === null || _a === void 0 || _a.register(viewDescriptor, pointer);
					value = emnapiCtx.addToCurrentScope(buffer_1).id;
					HEAP_DATA_VIEW.setUint32(result, value, true);
					HEAP_DATA_VIEW.setUint32(data, pointer, true);
				}
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ippppp
		*/
		function napi_create_buffer_copy(env, length, data, result_data, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				var Buffer$1 = emnapiCtx.feature.Buffer;
				if (!Buffer$1) throw emnapiCtx.createNotSupportBufferError("napi_create_buffer_copy", "");
				var arrayBuffer = emnapiCreateArrayBuffer(length, result_data);
				var buffer = Buffer$1.from(arrayBuffer);
				buffer.set(new Uint8Array(wasmMemory.buffer).subarray(data, data + length));
				value = emnapiCtx.addToCurrentScope(buffer).id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipppppp
		*/
		function napi_create_external_buffer(env, length, data, finalize_cb, finalize_hint, result) {
			return emnapi_create_memory_view(env, -2, data, length, finalize_cb, finalize_hint, result);
		}
		/**
		* @__sig ippppp
		*/
		function node_api_create_buffer_from_arraybuffer(env, arraybuffer, byte_offset, byte_length, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!arraybuffer) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				byte_offset = byte_offset >>> 0;
				byte_length = byte_length >>> 0;
				var handle = emnapiCtx.handleStore.get(arraybuffer);
				if (!handle.isArrayBuffer()) return envObject.setLastError(1);
				var buffer = handle.value;
				if (byte_length + byte_offset > buffer.byteLength) {
					var err$1 = /* @__PURE__ */ new RangeError("The byte offset + length is out of range");
					err$1.code = "ERR_OUT_OF_RANGE";
					throw err$1;
				}
				var Buffer$1 = emnapiCtx.feature.Buffer;
				if (!Buffer$1) throw emnapiCtx.createNotSupportBufferError("node_api_create_buffer_from_arraybuffer", "");
				var out = Buffer$1.from(buffer, byte_offset, byte_length);
				if (buffer === wasmMemory.buffer) {
					if (!emnapiExternalMemory.wasmMemoryViewTable.has(out)) emnapiExternalMemory.wasmMemoryViewTable.set(out, {
						Ctor: Buffer$1,
						address: byte_offset,
						length: byte_length,
						ownership: 1,
						runtimeAllocated: 0
					});
				}
				value = emnapiCtx.addToCurrentScope(out).id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$2) {
				envObject.tryCatch.setError(err$2);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ippppp
		*/
		function napi_create_dataview(env, byte_length, arraybuffer, byte_offset, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!arraybuffer) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				byte_length = byte_length >>> 0;
				byte_offset = byte_offset >>> 0;
				var handle = emnapiCtx.handleStore.get(arraybuffer);
				if (!handle.isArrayBuffer()) return envObject.setLastError(1);
				var buffer = handle.value;
				if (byte_length + byte_offset > buffer.byteLength) {
					var err$1 = /* @__PURE__ */ new RangeError("byte_offset + byte_length should be less than or equal to the size in bytes of the array passed in");
					err$1.code = "ERR_NAPI_INVALID_DATAVIEW_ARGS";
					throw err$1;
				}
				var dataview = new DataView(buffer, byte_offset, byte_length);
				if (buffer === wasmMemory.buffer) {
					if (!emnapiExternalMemory.wasmMemoryViewTable.has(dataview)) emnapiExternalMemory.wasmMemoryViewTable.set(dataview, {
						Ctor: DataView,
						address: byte_offset,
						length: byte_length,
						ownership: 1,
						runtimeAllocated: 0
					});
				}
				value = emnapiCtx.addToCurrentScope(dataview).id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$2) {
				envObject.tryCatch.setError(err$2);
				return envObject.setLastError(10);
			}
		}
		/**
		* @__sig ipppp
		*/
		function node_api_symbol_for(env, utf8description, length, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var autoLength = length === -1;
			var sizelength = length >>> 0;
			if (length !== 0) {
				if (!utf8description) return envObject.setLastError(1);
			}
			if (!(autoLength || sizelength <= 2147483647)) return envObject.setLastError(1);
			var descriptionString = emnapiString.UTF8ToString(utf8description, length);
			var value = emnapiCtx.addToCurrentScope(Symbol.for(descriptionString)).id;
			new DataView(wasmMemory.buffer).setUint32(result, value, true);
			return envObject.clearLastError();
		}
		var createMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_create_array,
			napi_create_array_with_length,
			napi_create_arraybuffer,
			napi_create_buffer,
			napi_create_buffer_copy,
			napi_create_dataview,
			napi_create_date,
			napi_create_external,
			napi_create_external_arraybuffer,
			napi_create_external_buffer,
			napi_create_object,
			napi_create_symbol,
			napi_create_typedarray,
			node_api_create_buffer_from_arraybuffer,
			node_api_symbol_for
		});
		/** @__sig ipip */
		function napi_get_boolean(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var v = value === 0 ? 3 : 4;
			new DataView(wasmMemory.buffer).setUint32(result, v, true);
			return envObject.clearLastError();
		}
		/** @__sig ipp */
		function napi_get_global(env, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			new DataView(wasmMemory.buffer).setUint32(result, 5, true);
			return envObject.clearLastError();
		}
		/** @__sig ipp */
		function napi_get_null(env, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			new DataView(wasmMemory.buffer).setUint32(result, 2, true);
			return envObject.clearLastError();
		}
		/** @__sig ipp */
		function napi_get_undefined(env, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			new DataView(wasmMemory.buffer).setUint32(result, 1, true);
			return envObject.clearLastError();
		}
		var globalMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_get_boolean,
			napi_get_global,
			napi_get_null,
			napi_get_undefined
		});
		/** @__sig ipppp */
		function napi_set_instance_data(env, data, finalize_cb, finalize_hint) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.setInstanceData(data, finalize_cb, finalize_hint);
			return envObject.clearLastError();
		}
		/** @__sig ipp */
		function napi_get_instance_data(env, data) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!data) return envObject.setLastError(1);
			var value = envObject.getInstanceData();
			new DataView(wasmMemory.buffer).setUint32(data, value, true);
			return envObject.clearLastError();
		}
		var envMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_get_instance_data,
			napi_set_instance_data
		});
		/** @__sig vpppp */
		function _emnapi_get_last_error_info(env, error_code, engine_error_code, engine_reserved) {
			var lastError = emnapiCtx.envStore.get(env).lastError;
			var errorCode = lastError.errorCode;
			var engineErrorCode = lastError.engineErrorCode >>> 0;
			var engineReserved = lastError.engineReserved;
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			HEAP_DATA_VIEW.setInt32(error_code, errorCode, true);
			HEAP_DATA_VIEW.setUint32(engine_error_code, engineErrorCode, true);
			HEAP_DATA_VIEW.setUint32(engine_reserved, engineReserved, true);
		}
		/** @__sig ipp */
		function napi_throw(env, error$1) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!error$1) return envObject.setLastError(1);
				envObject.tryCatch.setError(emnapiCtx.handleStore.get(error$1).value);
				return envObject.clearLastError();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_throw_error(env, code, msg) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!msg) return envObject.setLastError(1);
				var error$1 = new Error(emnapiString.UTF8ToString(msg, -1));
				if (code) error$1.code = emnapiString.UTF8ToString(code, -1);
				envObject.tryCatch.setError(error$1);
				return envObject.clearLastError();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_throw_type_error(env, code, msg) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!msg) return envObject.setLastError(1);
				var error$1 = new TypeError(emnapiString.UTF8ToString(msg, -1));
				if (code) error$1.code = emnapiString.UTF8ToString(code, -1);
				envObject.tryCatch.setError(error$1);
				return envObject.clearLastError();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_throw_range_error(env, code, msg) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!msg) return envObject.setLastError(1);
				var error$1 = new RangeError(emnapiString.UTF8ToString(msg, -1));
				if (code) error$1.code = emnapiString.UTF8ToString(code, -1);
				envObject.tryCatch.setError(error$1);
				return envObject.clearLastError();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function node_api_throw_syntax_error(env, code, msg) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!msg) return envObject.setLastError(1);
				var error$1 = new SyntaxError(emnapiString.UTF8ToString(msg, -1));
				if (code) error$1.code = emnapiString.UTF8ToString(code, -1);
				envObject.tryCatch.setError(error$1);
				return envObject.clearLastError();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipp */
		function napi_is_exception_pending(env, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var r = envObject.tryCatch.hasCaught();
			new DataView(wasmMemory.buffer).setInt8(result, r ? 1 : 0, true);
			return envObject.clearLastError();
		}
		/** @__sig ipppp */
		function napi_create_error(env, code, msg, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!msg) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var msgValue = emnapiCtx.handleStore.get(msg).value;
			if (typeof msgValue !== "string") return envObject.setLastError(3);
			var error$1 = new Error(msgValue);
			if (code) {
				var codeValue = emnapiCtx.handleStore.get(code).value;
				if (typeof codeValue !== "string") return envObject.setLastError(3);
				error$1.code = codeValue;
			}
			var value = emnapiCtx.addToCurrentScope(error$1).id;
			new DataView(wasmMemory.buffer).setUint32(result, value, true);
			return envObject.clearLastError();
		}
		/** @__sig ipppp */
		function napi_create_type_error(env, code, msg, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!msg) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var msgValue = emnapiCtx.handleStore.get(msg).value;
			if (typeof msgValue !== "string") return envObject.setLastError(3);
			var error$1 = new TypeError(msgValue);
			if (code) {
				var codeValue = emnapiCtx.handleStore.get(code).value;
				if (typeof codeValue !== "string") return envObject.setLastError(3);
				error$1.code = codeValue;
			}
			var value = emnapiCtx.addToCurrentScope(error$1).id;
			new DataView(wasmMemory.buffer).setUint32(result, value, true);
			return envObject.clearLastError();
		}
		/** @__sig ipppp */
		function napi_create_range_error(env, code, msg, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!msg) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var msgValue = emnapiCtx.handleStore.get(msg).value;
			if (typeof msgValue !== "string") return envObject.setLastError(3);
			var error$1 = new RangeError(msgValue);
			if (code) {
				var codeValue = emnapiCtx.handleStore.get(code).value;
				if (typeof codeValue !== "string") return envObject.setLastError(3);
				error$1.code = codeValue;
			}
			var value = emnapiCtx.addToCurrentScope(error$1).id;
			new DataView(wasmMemory.buffer).setUint32(result, value, true);
			return envObject.clearLastError();
		}
		/** @__sig ipppp */
		function node_api_create_syntax_error(env, code, msg, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!msg) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var msgValue = emnapiCtx.handleStore.get(msg).value;
			if (typeof msgValue !== "string") return envObject.setLastError(3);
			var error$1 = new SyntaxError(msgValue);
			if (code) {
				var codeValue = emnapiCtx.handleStore.get(code).value;
				if (typeof codeValue !== "string") return envObject.setLastError(3);
				error$1.code = codeValue;
			}
			var value = emnapiCtx.addToCurrentScope(error$1).id;
			new DataView(wasmMemory.buffer).setUint32(result, value, true);
			return envObject.clearLastError();
		}
		/** @__sig ipp */
		function napi_get_and_clear_last_exception(env, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (!envObject.tryCatch.hasCaught()) {
				HEAP_DATA_VIEW.setUint32(result, 1, true);
				return envObject.clearLastError();
			} else {
				var err$1 = envObject.tryCatch.exception();
				var value = envObject.ensureHandleId(err$1);
				HEAP_DATA_VIEW.setUint32(result, value, true);
				envObject.tryCatch.reset();
			}
			return envObject.clearLastError();
		}
		/** @__sig vpppp */
		function napi_fatal_error(location, location_len, message, message_len) {
			var locationStr = emnapiString.UTF8ToString(location, location_len);
			var messageStr = emnapiString.UTF8ToString(message, message_len);
			if (emnapiNodeBinding) emnapiNodeBinding.napi.fatalError(locationStr, messageStr);
			else abort("FATAL ERROR: " + locationStr + " " + messageStr);
		}
		/** @__sig ipp */
		function napi_fatal_exception(env, err$1) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!err$1) return envObject.setLastError(1);
				var error$1 = envObject.ctx.handleStore.get(err$1);
				try {
					envObject.triggerFatalException(error$1.value);
				} catch (_) {
					return envObject.setLastError(9);
				}
				return envObject.clearLastError();
			} catch (err$2) {
				envObject.tryCatch.setError(err$2);
				return envObject.setLastError(10);
			}
		}
		var errorMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			_emnapi_get_last_error_info,
			napi_create_error,
			napi_create_range_error,
			napi_create_type_error,
			napi_fatal_error,
			napi_fatal_exception,
			napi_get_and_clear_last_exception,
			napi_is_exception_pending,
			napi_throw,
			napi_throw_error,
			napi_throw_range_error,
			napi_throw_type_error,
			node_api_create_syntax_error,
			node_api_throw_syntax_error
		});
		/** @__sig ipppppp */
		function napi_create_function(env, utf8name, length, cb, data, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				if (!cb) return envObject.setLastError(1);
				var fresult = emnapiCreateFunction(envObject, utf8name, length, cb, data);
				if (fresult.status !== 0) return envObject.setLastError(fresult.status);
				var f = fresult.f;
				value = emnapiCtx.addToCurrentScope(f).id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipppppp */
		function napi_get_cb_info(env, cbinfo, argc, argv, this_arg, data) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!cbinfo) return envObject.setLastError(1);
			var cbinfoValue = emnapiCtx.scopeStore.get(cbinfo).callbackInfo;
			var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
			if (argv) {
				if (!argc) return envObject.setLastError(1);
				var argcValue = HEAP_DATA_VIEW.getUint32(argc, true);
				var len = cbinfoValue.args.length;
				var arrlen = argcValue < len ? argcValue : len;
				var i = 0;
				for (; i < arrlen; i++) {
					var argVal = envObject.ensureHandleId(cbinfoValue.args[i]);
					HEAP_DATA_VIEW.setUint32(argv + i * 4, argVal, true);
				}
				if (i < argcValue) for (; i < argcValue; i++) HEAP_DATA_VIEW.setUint32(argv + i * 4, 1, true);
			}
			if (argc) HEAP_DATA_VIEW.setUint32(argc, cbinfoValue.args.length, true);
			if (this_arg) {
				var v = envObject.ensureHandleId(cbinfoValue.thiz);
				HEAP_DATA_VIEW.setUint32(this_arg, v, true);
			}
			if (data) HEAP_DATA_VIEW.setUint32(data, cbinfoValue.data, true);
			return envObject.clearLastError();
		}
		/** @__sig ipppppp */
		function napi_call_function(env, recv, func, argc, argv, result) {
			var i = 0;
			var v;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!recv) return envObject.setLastError(1);
				argc = argc >>> 0;
				if (argc > 0) {
					if (!argv) return envObject.setLastError(1);
				}
				var v8recv = emnapiCtx.handleStore.get(recv).value;
				if (!func) return envObject.setLastError(1);
				var v8func = emnapiCtx.handleStore.get(func).value;
				if (typeof v8func !== "function") return envObject.setLastError(1);
				var args = [];
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				for (; i < argc; i++) {
					var argVal = HEAP_DATA_VIEW.getUint32(argv + i * 4, true);
					args.push(emnapiCtx.handleStore.get(argVal).value);
				}
				var ret = v8func.apply(v8recv, args);
				if (result) {
					v = envObject.ensureHandleId(ret);
					HEAP_DATA_VIEW.setUint32(result, v, true);
				}
				return envObject.clearLastError();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippppp */
		function napi_new_instance(env, constructor, argc, argv, result) {
			var i;
			var v;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!constructor) return envObject.setLastError(1);
				argc = argc >>> 0;
				if (argc > 0) {
					if (!argv) return envObject.setLastError(1);
				}
				if (!result) return envObject.setLastError(1);
				var Ctor = emnapiCtx.handleStore.get(constructor).value;
				if (typeof Ctor !== "function") return envObject.setLastError(1);
				var ret = void 0;
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				if (emnapiCtx.feature.supportReflect) {
					var argList = Array(argc);
					for (i = 0; i < argc; i++) {
						var argVal = HEAP_DATA_VIEW.getUint32(argv + i * 4, true);
						argList[i] = emnapiCtx.handleStore.get(argVal).value;
					}
					ret = Reflect.construct(Ctor, argList, Ctor);
				} else {
					var args = Array(argc + 1);
					args[0] = void 0;
					for (i = 0; i < argc; i++) {
						var argVal = HEAP_DATA_VIEW.getUint32(argv + i * 4, true);
						args[i + 1] = emnapiCtx.handleStore.get(argVal).value;
					}
					ret = new (Ctor.bind.apply(Ctor, args))();
				}
				if (result) {
					v = envObject.ensureHandleId(ret);
					HEAP_DATA_VIEW.setUint32(result, v, true);
				}
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_get_new_target(env, cbinfo, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!cbinfo) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var cbinfoValue = emnapiCtx.scopeStore.get(cbinfo).callbackInfo;
			var thiz = cbinfoValue.thiz, fn = cbinfoValue.fn;
			var value = thiz == null || thiz.constructor == null ? 0 : thiz instanceof fn ? envObject.ensureHandleId(thiz.constructor) : 0;
			new DataView(wasmMemory.buffer).setUint32(result, value, true);
			return envObject.clearLastError();
		}
		var functionMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_call_function,
			napi_create_function,
			napi_get_cb_info,
			napi_get_new_target,
			napi_new_instance
		});
		/** @__sig ipp */
		function napi_open_handle_scope(env, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var scope = emnapiCtx.openScope(envObject);
			new DataView(wasmMemory.buffer).setUint32(result, scope.id, true);
			return envObject.clearLastError();
		}
		/** @__sig ipp */
		function napi_close_handle_scope(env, scope) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!scope) return envObject.setLastError(1);
			if (envObject.openHandleScopes === 0) return 13;
			emnapiCtx.closeScope(envObject);
			return envObject.clearLastError();
		}
		/** @__sig ipp */
		function napi_open_escapable_handle_scope(env, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!result) return envObject.setLastError(1);
			var scope = emnapiCtx.openScope(envObject);
			new DataView(wasmMemory.buffer).setUint32(result, scope.id, true);
			return envObject.clearLastError();
		}
		/** @__sig ipp */
		function napi_close_escapable_handle_scope(env, scope) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!scope) return envObject.setLastError(1);
			if (envObject.openHandleScopes === 0) return 13;
			emnapiCtx.closeScope(envObject);
			return envObject.clearLastError();
		}
		/** @__sig ipppp */
		function napi_escape_handle(env, scope, escapee, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!scope) return envObject.setLastError(1);
			if (!escapee) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var scopeObject = emnapiCtx.scopeStore.get(scope);
			if (!scopeObject.escapeCalled()) {
				var newHandle = scopeObject.escape(escapee);
				var value = newHandle ? newHandle.id : 0;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.clearLastError();
			}
			return envObject.setLastError(12);
		}
		/** @__sig ippip */
		function napi_create_reference(env, value, initial_refcount, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var handle = emnapiCtx.handleStore.get(value);
			if (envObject.moduleApiVersion < 10) {
				if (!(handle.isObject() || handle.isFunction() || handle.isSymbol())) return envObject.setLastError(1);
			}
			var ref = emnapiCtx.createReference(envObject, handle.id, initial_refcount >>> 0, 1);
			new DataView(wasmMemory.buffer).setUint32(result, ref.id, true);
			return envObject.clearLastError();
		}
		/** @__sig ipp */
		function napi_delete_reference(env, ref) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!ref) return envObject.setLastError(1);
			emnapiCtx.refStore.get(ref).dispose();
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_reference_ref(env, ref, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!ref) return envObject.setLastError(1);
			var count = emnapiCtx.refStore.get(ref).ref();
			if (result) new DataView(wasmMemory.buffer).setUint32(result, count, true);
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_reference_unref(env, ref, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!ref) return envObject.setLastError(1);
			var reference = emnapiCtx.refStore.get(ref);
			if (reference.refcount() === 0) return envObject.setLastError(9);
			var count = reference.unref();
			if (result) new DataView(wasmMemory.buffer).setUint32(result, count, true);
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_get_reference_value(env, ref, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!ref) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var handleId = emnapiCtx.refStore.get(ref).get(envObject);
			new DataView(wasmMemory.buffer).setUint32(result, handleId, true);
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_add_env_cleanup_hook(env, fun, arg) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!fun) return envObject.setLastError(1);
			emnapiCtx.addCleanupHook(envObject, fun, arg);
			return 0;
		}
		/** @__sig ippp */
		function napi_remove_env_cleanup_hook(env, fun, arg) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!fun) return envObject.setLastError(1);
			emnapiCtx.removeCleanupHook(envObject, fun, arg);
			return 0;
		}
		/** @__sig vp */
		function _emnapi_env_ref(env) {
			emnapiCtx.envStore.get(env).ref();
		}
		/** @__sig vp */
		function _emnapi_env_unref(env) {
			emnapiCtx.envStore.get(env).unref();
		}
		var lifeMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			_emnapi_env_ref,
			_emnapi_env_unref,
			napi_add_env_cleanup_hook,
			napi_close_escapable_handle_scope,
			napi_close_handle_scope,
			napi_create_reference,
			napi_delete_reference,
			napi_escape_handle,
			napi_get_reference_value,
			napi_open_escapable_handle_scope,
			napi_open_handle_scope,
			napi_reference_ref,
			napi_reference_unref,
			napi_remove_env_cleanup_hook
		});
		/** @__sig ippi */
		function _emnapi_get_filename(env, buf, len) {
			var filename = emnapiCtx.envStore.get(env).filename;
			if (!buf) return emnapiString.lengthBytesUTF8(filename);
			return emnapiString.stringToUTF8(filename, buf, len);
		}
		var miscellaneousMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			_emnapi_get_filename
		});
		/** @__sig ippp */
		function napi_create_promise(env, deferred, promise) {
			var deferredObjectId, value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!deferred) return envObject.setLastError(1);
				if (!promise) return envObject.setLastError(1);
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				var p = new Promise(function(resolve$1, reject) {
					deferredObjectId = emnapiCtx.createDeferred({
						resolve: resolve$1,
						reject
					}).id;
					HEAP_DATA_VIEW.setUint32(deferred, deferredObjectId, true);
				});
				value = emnapiCtx.addToCurrentScope(p).id;
				HEAP_DATA_VIEW.setUint32(promise, value, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_resolve_deferred(env, deferred, resolution) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!deferred) return envObject.setLastError(1);
				if (!resolution) return envObject.setLastError(1);
				emnapiCtx.deferredStore.get(deferred).resolve(emnapiCtx.handleStore.get(resolution).value);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_reject_deferred(env, deferred, resolution) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!deferred) return envObject.setLastError(1);
				if (!resolution) return envObject.setLastError(1);
				emnapiCtx.deferredStore.get(deferred).reject(emnapiCtx.handleStore.get(resolution).value);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_is_promise(env, value, is_promise) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!is_promise) return envObject.setLastError(1);
			var r = emnapiCtx.handleStore.get(value).isPromise() ? 1 : 0;
			new DataView(wasmMemory.buffer).setInt8(is_promise, r, true);
			return envObject.clearLastError();
		}
		var promiseMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_create_promise,
			napi_is_promise,
			napi_reject_deferred,
			napi_resolve_deferred
		});
		/** @__sig ippiiip */
		function napi_get_all_property_names(env, object, key_mode, key_filter, key_conversion, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (h.value == null) throw new TypeError("Cannot convert undefined or null to object");
				var obj = void 0;
				try {
					obj = h.isObject() || h.isFunction() ? h.value : Object(h.value);
				} catch (_) {
					return envObject.setLastError(2);
				}
				if (key_mode !== 0 && key_mode !== 1) return envObject.setLastError(1);
				if (key_conversion !== 0 && key_conversion !== 1) return envObject.setLastError(1);
				var props = [];
				var names = void 0;
				var symbols = void 0;
				var i = void 0;
				var own = true;
				var integerIndiceRegex = /^(0|[1-9][0-9]*)$/;
				do {
					names = Object.getOwnPropertyNames(obj);
					symbols = Object.getOwnPropertySymbols(obj);
					for (i = 0; i < names.length; i++) props.push({
						name: integerIndiceRegex.test(names[i]) ? Number(names[i]) : names[i],
						desc: Object.getOwnPropertyDescriptor(obj, names[i]),
						own
					});
					for (i = 0; i < symbols.length; i++) props.push({
						name: symbols[i],
						desc: Object.getOwnPropertyDescriptor(obj, symbols[i]),
						own
					});
					if (key_mode === 1) break;
					obj = Object.getPrototypeOf(obj);
					own = false;
				} while (obj);
				var ret = [];
				var addName = function(ret$1, name, key_filter$1, conversion_mode) {
					if (ret$1.indexOf(name) !== -1) return;
					if (conversion_mode === 0) ret$1.push(name);
					else if (conversion_mode === 1) {
						var realName = typeof name === "number" ? String(name) : name;
						if (typeof realName === "string") {
							if (!(key_filter$1 & 8)) ret$1.push(realName);
						} else ret$1.push(realName);
					}
				};
				for (i = 0; i < props.length; i++) {
					var prop = props[i];
					var name_1 = prop.name;
					var desc = prop.desc;
					if (key_filter === 0) addName(ret, name_1, key_filter, key_conversion);
					else {
						if (key_filter & 8 && typeof name_1 === "string") continue;
						if (key_filter & 16 && typeof name_1 === "symbol") continue;
						var shouldAdd = true;
						switch (key_filter & 7) {
							case 1:
								shouldAdd = Boolean(desc.writable);
								break;
							case 2:
								shouldAdd = Boolean(desc.enumerable);
								break;
							case 3:
								shouldAdd = Boolean(desc.writable && desc.enumerable);
								break;
							case 4:
								shouldAdd = Boolean(desc.configurable);
								break;
							case 5:
								shouldAdd = Boolean(desc.configurable && desc.writable);
								break;
							case 6:
								shouldAdd = Boolean(desc.configurable && desc.enumerable);
								break;
							case 7:
								shouldAdd = Boolean(desc.configurable && desc.enumerable && desc.writable);
								break;
						}
						if (shouldAdd) addName(ret, name_1, key_filter, key_conversion);
					}
				}
				value = emnapiCtx.addToCurrentScope(ret).id;
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_get_property_names(env, object, result) {
			return napi_get_all_property_names(env, object, 0, 18, 1, result);
		}
		/** @__sig ipppp */
		function napi_set_property(env, object, key, value) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!key) return envObject.setLastError(1);
				if (!value) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (!(h.isObject() || h.isFunction())) return envObject.setLastError(2);
				h.value[emnapiCtx.handleStore.get(key).value] = emnapiCtx.handleStore.get(value).value;
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipppp */
		function napi_has_property(env, object, key, result) {
			var r;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!key) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (h.value == null) throw new TypeError("Cannot convert undefined or null to object");
				var v = void 0;
				try {
					v = h.isObject() || h.isFunction() ? h.value : Object(h.value);
				} catch (_) {
					return envObject.setLastError(2);
				}
				r = emnapiCtx.handleStore.get(key).value in v ? 1 : 0;
				new DataView(wasmMemory.buffer).setInt8(result, r, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipppp */
		function napi_get_property(env, object, key, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!key) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (h.value == null) throw new TypeError("Cannot convert undefined or null to object");
				var v = void 0;
				try {
					v = h.isObject() || h.isFunction() ? h.value : Object(h.value);
				} catch (_) {
					return envObject.setLastError(2);
				}
				value = envObject.ensureHandleId(v[emnapiCtx.handleStore.get(key).value]);
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipppp */
		function napi_delete_property(env, object, key, result) {
			var r;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!key) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (!(h.isObject() || h.isFunction())) return envObject.setLastError(2);
				var propertyKey = emnapiCtx.handleStore.get(key).value;
				if (emnapiCtx.feature.supportReflect) r = Reflect.deleteProperty(h.value, propertyKey);
				else try {
					r = delete h.value[propertyKey];
				} catch (_) {
					r = false;
				}
				if (result) new DataView(wasmMemory.buffer).setInt8(result, r ? 1 : 0, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipppp */
		function napi_has_own_property(env, object, key, result) {
			var r;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!key) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (h.value == null) throw new TypeError("Cannot convert undefined or null to object");
				var v = void 0;
				try {
					v = h.isObject() || h.isFunction() ? h.value : Object(h.value);
				} catch (_) {
					return envObject.setLastError(2);
				}
				var prop = emnapiCtx.handleStore.get(key).value;
				if (typeof prop !== "string" && typeof prop !== "symbol") return envObject.setLastError(4);
				r = Object.prototype.hasOwnProperty.call(v, emnapiCtx.handleStore.get(key).value);
				new DataView(wasmMemory.buffer).setInt8(result, r ? 1 : 0, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipppp */
		function napi_set_named_property(env, object, cname, value) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!value) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (!(h.isObject() || h.isFunction())) return envObject.setLastError(2);
				if (!cname) return envObject.setLastError(1);
				emnapiCtx.handleStore.get(object).value[emnapiString.UTF8ToString(cname, -1)] = emnapiCtx.handleStore.get(value).value;
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipppp */
		function napi_has_named_property(env, object, utf8name, result) {
			var r;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				if (!utf8name) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (h.value == null) throw new TypeError("Cannot convert undefined or null to object");
				var v = void 0;
				try {
					v = h.isObject() || h.isFunction() ? h.value : Object(h.value);
				} catch (_) {
					return envObject.setLastError(2);
				}
				r = emnapiString.UTF8ToString(utf8name, -1) in v;
				new DataView(wasmMemory.buffer).setInt8(result, r ? 1 : 0, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipppp */
		function napi_get_named_property(env, object, utf8name, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				if (!utf8name) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (h.value == null) throw new TypeError("Cannot convert undefined or null to object");
				var v = void 0;
				try {
					v = h.isObject() || h.isFunction() ? h.value : Object(h.value);
				} catch (_) {
					return envObject.setLastError(2);
				}
				value = envObject.ensureHandleId(v[emnapiString.UTF8ToString(utf8name, -1)]);
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippip */
		function napi_set_element(env, object, index, value) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!value) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (!(h.isObject() || h.isFunction())) return envObject.setLastError(2);
				h.value[index >>> 0] = emnapiCtx.handleStore.get(value).value;
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippip */
		function napi_has_element(env, object, index, result) {
			var r;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (h.value == null) throw new TypeError("Cannot convert undefined or null to object");
				var v = void 0;
				try {
					v = h.isObject() || h.isFunction() ? h.value : Object(h.value);
				} catch (_) {
					return envObject.setLastError(2);
				}
				r = index >>> 0 in v ? 1 : 0;
				new DataView(wasmMemory.buffer).setInt8(result, r, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippip */
		function napi_get_element(env, object, index, result) {
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!result) return envObject.setLastError(1);
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (h.value == null) throw new TypeError("Cannot convert undefined or null to object");
				var v = void 0;
				try {
					v = h.isObject() || h.isFunction() ? h.value : Object(h.value);
				} catch (_) {
					return envObject.setLastError(2);
				}
				value = envObject.ensureHandleId(v[index >>> 0]);
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippip */
		function napi_delete_element(env, object, index, result) {
			var r;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				if (!(h.isObject() || h.isFunction())) return envObject.setLastError(2);
				if (emnapiCtx.feature.supportReflect) r = Reflect.deleteProperty(h.value, index >>> 0);
				else try {
					r = delete h.value[index >>> 0];
				} catch (_) {
					r = false;
				}
				if (result) new DataView(wasmMemory.buffer).setInt8(result, r ? 1 : 0, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipppp */
		function napi_define_properties(env, object, property_count, properties) {
			var propPtr, attributes;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				property_count = property_count >>> 0;
				if (property_count > 0) {
					if (!properties) return envObject.setLastError(1);
				}
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				var maybeObject = h.value;
				if (!(h.isObject() || h.isFunction())) return envObject.setLastError(2);
				var propertyName = void 0;
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				for (var i = 0; i < property_count; i++) {
					propPtr = properties + i * 32;
					var utf8Name = HEAP_DATA_VIEW.getUint32(propPtr, true);
					var name_2 = HEAP_DATA_VIEW.getUint32(propPtr + 4, true);
					var method = HEAP_DATA_VIEW.getUint32(propPtr + 8, true);
					var getter = HEAP_DATA_VIEW.getUint32(propPtr + 12, true);
					var setter = HEAP_DATA_VIEW.getUint32(propPtr + 16, true);
					var value = HEAP_DATA_VIEW.getUint32(propPtr + 20, true);
					attributes = HEAP_DATA_VIEW.getInt32(propPtr + 24, true);
					var data = HEAP_DATA_VIEW.getUint32(propPtr + 28, true);
					if (utf8Name) propertyName = emnapiString.UTF8ToString(utf8Name, -1);
					else {
						if (!name_2) return envObject.setLastError(4);
						propertyName = emnapiCtx.handleStore.get(name_2).value;
						if (typeof propertyName !== "string" && typeof propertyName !== "symbol") return envObject.setLastError(4);
					}
					emnapiDefineProperty(envObject, maybeObject, propertyName, method, getter, setter, value, attributes, data);
				}
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipp */
		function napi_object_freeze(env, object) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				var maybeObject = h.value;
				if (!(h.isObject() || h.isFunction())) return envObject.setLastError(2);
				Object.freeze(maybeObject);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipp */
		function napi_object_seal(env, object) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!object) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(object);
				var maybeObject = h.value;
				if (!(h.isObject() || h.isFunction())) return envObject.setLastError(2);
				Object.seal(maybeObject);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		var propertyMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_define_properties,
			napi_delete_element,
			napi_delete_property,
			napi_get_all_property_names,
			napi_get_element,
			napi_get_named_property,
			napi_get_property,
			napi_get_property_names,
			napi_has_element,
			napi_has_named_property,
			napi_has_own_property,
			napi_has_property,
			napi_object_freeze,
			napi_object_seal,
			napi_set_element,
			napi_set_named_property,
			napi_set_property
		});
		/** @__sig ippp */
		function napi_run_script(env, script, result) {
			var status;
			var value;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!script) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				var v8Script = emnapiCtx.handleStore.get(script);
				if (!v8Script.isString()) return envObject.setLastError(3);
				var ret = emnapiCtx.handleStore.get(5).value.eval(v8Script.value);
				value = envObject.ensureHandleId(ret);
				new DataView(wasmMemory.buffer).setUint32(result, value, true);
				status = envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
			return status;
		}
		var scriptMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_run_script
		});
		/** @__sig ippp */
		function napi_typeof(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var v = emnapiCtx.handleStore.get(value);
			var r;
			if (v.isNumber()) r = 3;
			else if (v.isBigInt()) r = 9;
			else if (v.isString()) r = 4;
			else if (v.isFunction()) r = 7;
			else if (v.isExternal()) r = 8;
			else if (v.isObject()) r = 6;
			else if (v.isBoolean()) r = 2;
			else if (v.isUndefined()) r = 0;
			else if (v.isSymbol()) r = 5;
			else if (v.isNull()) r = 1;
			else return envObject.setLastError(1);
			new DataView(wasmMemory.buffer).setInt32(result, r, true);
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_coerce_to_bool(env, value, result) {
			var v;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!value) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				v = emnapiCtx.handleStore.get(value).value ? 4 : 3;
				new DataView(wasmMemory.buffer).setUint32(result, v, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_coerce_to_number(env, value, result) {
			var v;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!value) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				var handle = emnapiCtx.handleStore.get(value);
				if (handle.isBigInt()) throw new TypeError("Cannot convert a BigInt value to a number");
				v = emnapiCtx.addToCurrentScope(Number(handle.value)).id;
				new DataView(wasmMemory.buffer).setUint32(result, v, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_coerce_to_object(env, value, result) {
			var v;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!value) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				var handle = emnapiCtx.handleStore.get(value);
				if (handle.value == null) throw new TypeError("Cannot convert undefined or null to object");
				v = envObject.ensureHandleId(Object(handle.value));
				new DataView(wasmMemory.buffer).setUint32(result, v, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_coerce_to_string(env, value, result) {
			var v;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!value) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				var handle = emnapiCtx.handleStore.get(value);
				if (handle.isSymbol()) throw new TypeError("Cannot convert a Symbol value to a string");
				v = emnapiCtx.addToCurrentScope(String(handle.value)).id;
				new DataView(wasmMemory.buffer).setUint32(result, v, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipppp */
		function napi_instanceof(env, object, constructor, result) {
			var r;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!object) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				if (!constructor) return envObject.setLastError(1);
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				HEAP_DATA_VIEW.setInt8(result, 0, true);
				var ctor = emnapiCtx.handleStore.get(constructor);
				if (!ctor.isFunction()) return envObject.setLastError(5);
				r = emnapiCtx.handleStore.get(object).value instanceof ctor.value ? 1 : 0;
				HEAP_DATA_VIEW.setInt8(result, r, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ippp */
		function napi_is_array(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var r = emnapiCtx.handleStore.get(value).isArray() ? 1 : 0;
			new DataView(wasmMemory.buffer).setInt8(result, r, true);
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_is_arraybuffer(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var r = emnapiCtx.handleStore.get(value).isArrayBuffer() ? 1 : 0;
			new DataView(wasmMemory.buffer).setInt8(result, r, true);
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_is_date(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var r = emnapiCtx.handleStore.get(value).isDate() ? 1 : 0;
			new DataView(wasmMemory.buffer).setInt8(result, r, true);
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_is_error(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var r = emnapiCtx.handleStore.get(value).value instanceof Error ? 1 : 0;
			new DataView(wasmMemory.buffer).setInt8(result, r, true);
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_is_typedarray(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var r = emnapiCtx.handleStore.get(value).isTypedArray() ? 1 : 0;
			new DataView(wasmMemory.buffer).setInt8(result, r, true);
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_is_buffer(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var r = emnapiCtx.handleStore.get(value).isBuffer(emnapiCtx.feature.Buffer) ? 1 : 0;
			new DataView(wasmMemory.buffer).setInt8(result, r, true);
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_is_dataview(env, value, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!value) return envObject.setLastError(1);
			if (!result) return envObject.setLastError(1);
			var r = emnapiCtx.handleStore.get(value).isDataView() ? 1 : 0;
			new DataView(wasmMemory.buffer).setInt8(result, r, true);
			return envObject.clearLastError();
		}
		/** @__sig ipppp */
		function napi_strict_equals(env, lhs, rhs, result) {
			var r;
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!lhs) return envObject.setLastError(1);
				if (!rhs) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				var lv = emnapiCtx.handleStore.get(lhs).value;
				var rv = emnapiCtx.handleStore.get(rhs).value;
				r = lv === rv ? 1 : 0;
				new DataView(wasmMemory.buffer).setInt8(result, r, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		/** @__sig ipp */
		function napi_detach_arraybuffer(env, arraybuffer) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!arraybuffer) return envObject.setLastError(1);
			var value = emnapiCtx.handleStore.get(arraybuffer).value;
			if (!(value instanceof ArrayBuffer)) {
				if (typeof SharedArrayBuffer === "function" && value instanceof SharedArrayBuffer) return envObject.setLastError(20);
				return envObject.setLastError(19);
			}
			try {
				var MessageChannel_1 = emnapiCtx.feature.MessageChannel;
				new MessageChannel_1().port1.postMessage(value, [value]);
			} catch (_) {
				return envObject.setLastError(9);
			}
			return envObject.clearLastError();
		}
		/** @__sig ippp */
		function napi_is_detached_arraybuffer(env, arraybuffer, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			envObject.checkGCAccess();
			if (!envObject.tryCatch.isEmpty()) return envObject.setLastError(10);
			if (!envObject.canCallIntoJs()) return envObject.setLastError(envObject.moduleApiVersion >= 10 ? 23 : 10);
			envObject.clearLastError();
			try {
				if (!arraybuffer) return envObject.setLastError(1);
				if (!result) return envObject.setLastError(1);
				var h = emnapiCtx.handleStore.get(arraybuffer);
				var HEAP_DATA_VIEW = new DataView(wasmMemory.buffer);
				if (h.isArrayBuffer() && h.value.byteLength === 0) try {
					new Uint8Array(h.value);
				} catch (_) {
					HEAP_DATA_VIEW.setInt8(result, 1, true);
					return envObject.getReturnStatus();
				}
				HEAP_DATA_VIEW.setInt8(result, 0, true);
				return envObject.getReturnStatus();
			} catch (err$1) {
				envObject.tryCatch.setError(err$1);
				return envObject.setLastError(10);
			}
		}
		var valueOperationMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_coerce_to_bool,
			napi_coerce_to_number,
			napi_coerce_to_object,
			napi_coerce_to_string,
			napi_detach_arraybuffer,
			napi_instanceof,
			napi_is_array,
			napi_is_arraybuffer,
			napi_is_buffer,
			napi_is_dataview,
			napi_is_date,
			napi_is_detached_arraybuffer,
			napi_is_error,
			napi_is_typedarray,
			napi_strict_equals,
			napi_typeof
		});
		/** @__sig ipp */
		function napi_get_version(env, result) {
			if (!env) return 1;
			var envObject = emnapiCtx.envStore.get(env);
			if (!result) return envObject.setLastError(1);
			new DataView(wasmMemory.buffer).setUint32(result, 10, true);
			return envObject.clearLastError();
		}
		var versionMod = /* @__PURE__ */ Object.freeze({
			__proto__: null,
			napi_get_version
		});
		emnapiAWST.init();
		emnapiExternalMemory.init();
		emnapiString.init();
		emnapiTSFN.init();
		PThread.init();
		napiModule.emnapi.syncMemory = $emnapiSyncMemory;
		napiModule.emnapi.getMemoryAddress = $emnapiGetMemoryAddress;
		function addImports(mod) {
			var keys = Object.keys(mod);
			for (var i = 0; i < keys.length; ++i) {
				var k = keys[i];
				if (k.indexOf("$") === 0) continue;
				if (k.indexOf("emnapi_") === 0) napiModule.imports.emnapi[k] = mod[k];
				else if (k.indexOf("_emnapi_") === 0 || k === "napi_set_last_error" || k === "napi_clear_last_error") napiModule.imports.env[k] = mod[k];
				else napiModule.imports.napi[k] = mod[k];
			}
		}
		addImports(asyncMod);
		addImports(memoryMod);
		addImports(asyncWorkMod);
		addImports(utilMod);
		addImports(convert2cMod);
		addImports(convert2napiMod);
		addImports(createMod);
		addImports(globalMod);
		addImports(wrapMod);
		addImports(envMod);
		addImports(emnapiMod);
		addImports(errorMod);
		addImports(functionMod);
		addImports(lifeMod);
		addImports(miscellaneousMod);
		addImports(nodeMod);
		addImports(promiseMod);
		addImports(propertyMod);
		addImports(scriptMod);
		addImports(valueOperationMod);
		addImports(versionMod);
		napiModule.imports.napi.napi_create_threadsafe_function = napi_create_threadsafe_function;
		napiModule.imports.napi.napi_get_threadsafe_function_context = napi_get_threadsafe_function_context;
		napiModule.imports.napi.napi_call_threadsafe_function = napi_call_threadsafe_function;
		napiModule.imports.napi.napi_acquire_threadsafe_function = napi_acquire_threadsafe_function;
		napiModule.imports.napi.napi_release_threadsafe_function = napi_release_threadsafe_function;
		napiModule.imports.napi.napi_unref_threadsafe_function = napi_unref_threadsafe_function;
		napiModule.imports.napi.napi_ref_threadsafe_function = napi_ref_threadsafe_function;
		return napiModule;
	})();
}
function loadNapiModuleImpl(loadFn, userNapiModule, wasmInput, options) {
	options = options !== null && options !== void 0 ? options : {};
	var getMemory$1 = options.getMemory;
	var getTable = options.getTable;
	var beforeInit = options.beforeInit;
	if (getMemory$1 != null && typeof getMemory$1 !== "function") throw new TypeError("options.getMemory is not a function");
	if (getTable != null && typeof getTable !== "function") throw new TypeError("options.getTable is not a function");
	if (beforeInit != null && typeof beforeInit !== "function") throw new TypeError("options.beforeInit is not a function");
	var napiModule;
	var isLoad = typeof userNapiModule === "object" && userNapiModule !== null;
	if (isLoad) {
		if (userNapiModule.loaded) throw new Error("napiModule has already loaded");
		napiModule = userNapiModule;
	} else napiModule = createNapiModule(options);
	var wasi = options.wasi;
	var wasiThreads;
	var importObject = {
		env: napiModule.imports.env,
		napi: napiModule.imports.napi,
		emnapi: napiModule.imports.emnapi
	};
	if (wasi) {
		wasiThreads = new WASIThreads(napiModule.childThread ? {
			wasi,
			childThread: true,
			postMessage: napiModule.postMessage
		} : {
			wasi,
			threadManager: napiModule.PThread,
			waitThreadStart: napiModule.waitThreadStart
		});
		Object.assign(importObject, typeof wasi.getImportObject === "function" ? wasi.getImportObject() : { wasi_snapshot_preview1: wasi.wasiImport });
		Object.assign(importObject, wasiThreads.getImportObject());
	}
	var overwriteImports = options.overwriteImports;
	if (typeof overwriteImports === "function") {
		var newImportObject = overwriteImports(importObject);
		if (typeof newImportObject === "object" && newImportObject !== null) importObject = newImportObject;
	}
	return loadFn(wasmInput, importObject, function(err, source) {
		if (err) throw err;
		var originalInstance = source.instance;
		var instance = originalInstance;
		var originalExports = originalInstance.exports;
		var exportMemory = "memory" in originalExports;
		var importMemory = "memory" in importObject.env;
		var memory = getMemory$1 ? getMemory$1(originalExports) : exportMemory ? originalExports.memory : importMemory ? importObject.env.memory : void 0;
		if (!memory) throw new Error("memory is neither exported nor imported");
		var table = getTable ? getTable(originalExports) : originalExports.__indirect_function_table;
		if (wasi && !exportMemory) {
			var exports_1 = Object.create(null);
			Object.assign(exports_1, originalExports, { memory });
			instance = { exports: exports_1 };
		}
		var module$1 = source.module;
		if (wasi) instance = wasiThreads.initialize(instance, module$1, memory);
		else napiModule.PThread.setup(module$1, memory);
		var emnapiInit = function() {
			if (beforeInit) beforeInit({
				instance: originalInstance,
				module: module$1
			});
			napiModule.init({
				instance,
				module: module$1,
				memory,
				table
			});
			var ret = {
				instance: originalInstance,
				module: module$1,
				usedInstance: instance
			};
			if (!isLoad) ret.napiModule = napiModule;
			return ret;
		};
		if (napiModule.PThread.shouldPreloadWorkers()) {
			var poolReady = napiModule.PThread.loadWasmModuleToAllWorkers();
			if (loadFn === loadCallback) return poolReady.then(emnapiInit);
			else throw new Error("Synchronous loading is not supported with worker pool (reuseWorker.size > 0)");
		}
		return emnapiInit();
	});
}
function loadCallback(wasmInput, importObject, callback) {
	return load$1(wasmInput, importObject).then(function(source) {
		return callback(null, source);
	}, function(err) {
		return callback(err);
	});
}
function loadSyncCallback(wasmInput, importObject, callback) {
	var source;
	try {
		source = loadSync$1(wasmInput, importObject);
	} catch (err) {
		return callback(err);
	}
	return callback(null, source);
}
/** @public */
function loadNapiModule(napiModule, wasmInput, options) {
	if (typeof napiModule !== "object" || napiModule === null) throw new TypeError("Invalid napiModule");
	return loadNapiModuleImpl(loadCallback, napiModule, wasmInput, options);
}
/** @public */
function loadNapiModuleSync(napiModule, wasmInput, options) {
	if (typeof napiModule !== "object" || napiModule === null) throw new TypeError("Invalid napiModule");
	return loadNapiModuleImpl(loadSyncCallback, napiModule, wasmInput, options);
}
/** @public */
function instantiateNapiModule$1(wasmInput, options) {
	return loadNapiModuleImpl(loadCallback, void 0, wasmInput, options);
}
/** @public */
function instantiateNapiModuleSync$1(wasmInput, options) {
	return loadNapiModuleImpl(loadSyncCallback, void 0, wasmInput, options);
}
var _WebAssembly$1, MessageHandler$1, version$2;
var init_emnapi_core_esm_bundler = __esm({ "../../node_modules/.pnpm/@emnapi+core@1.5.0/node_modules/@emnapi/core/dist/emnapi-core.esm-bundler.js": (() => {
	init_wasi_threads_esm_bundler();
	init_wasi_threads_esm_bundler();
	init_tslib_es6();
	_WebAssembly$1 = typeof WebAssembly !== "undefined" ? WebAssembly : typeof WXWebAssembly !== "undefined" ? WXWebAssembly : void 0;
	MessageHandler$1 = /* @__PURE__ */ function(_super) {
		__extends(MessageHandler$2, _super);
		function MessageHandler$2(options) {
			var _this = this;
			if (typeof options.onLoad !== "function") throw new TypeError("options.onLoad is not a function");
			var userOnError = options.onError;
			_this = _super.call(this, __assign(__assign({}, options), { onError: function(err, type) {
				var _a;
				var emnapi_thread_crashed = (_a = _this.instance) === null || _a === void 0 ? void 0 : _a.exports.emnapi_thread_crashed;
				if (typeof emnapi_thread_crashed === "function") emnapi_thread_crashed();
				if (typeof userOnError === "function") userOnError(err, type);
				else throw err;
			} })) || this;
			_this.napiModule = void 0;
			return _this;
		}
		MessageHandler$2.prototype.instantiate = function(data) {
			var _this = this;
			var source = this.onLoad(data);
			if (typeof source.then === "function") return source.then(function(result) {
				_this.napiModule = result.napiModule;
				return result;
			});
			this.napiModule = source.napiModule;
			return source;
		};
		MessageHandler$2.prototype.handle = function(e) {
			var _this = this;
			var _a;
			_super.prototype.handle.call(this, e);
			if ((_a = e === null || e === void 0 ? void 0 : e.data) === null || _a === void 0 ? void 0 : _a.__emnapi__) {
				var type = e.data.__emnapi__.type;
				var payload_1 = e.data.__emnapi__.payload;
				try {
					if (type === "async-worker-init") this.handleAfterLoad(e, function() {
						_this.napiModule.initWorker(payload_1.arg);
					});
					else if (type === "async-work-execute") this.handleAfterLoad(e, function() {
						_this.napiModule.executeAsyncWork(payload_1.work);
					});
				} catch (err) {
					this.onError(err, type);
				}
			}
		};
		return MessageHandler$2;
	}(ThreadMessageHandler);
	version$2 = "1.5.0";
}) });

//#endregion
//#region ../../node_modules/.pnpm/@emnapi+runtime@1.5.0/node_modules/@emnapi/runtime/dist/emnapi.esm-bundler.js
var emnapi_esm_bundler_exports = /* @__PURE__ */ __export({
	ConstHandle: () => ConstHandle,
	Context: () => Context,
	Deferred: () => Deferred,
	EmnapiError: () => EmnapiError,
	Env: () => Env,
	External: () => External,
	Finalizer: () => Finalizer,
	Handle: () => Handle,
	HandleScope: () => HandleScope,
	HandleStore: () => HandleStore,
	NAPI_VERSION_EXPERIMENTAL: () => NAPI_VERSION_EXPERIMENTAL,
	NODE_API_DEFAULT_MODULE_API_VERSION: () => NODE_API_DEFAULT_MODULE_API_VERSION,
	NODE_API_SUPPORTED_VERSION_MAX: () => NODE_API_SUPPORTED_VERSION_MAX,
	NODE_API_SUPPORTED_VERSION_MIN: () => NODE_API_SUPPORTED_VERSION_MIN,
	NodeEnv: () => NodeEnv,
	NotSupportBufferError: () => NotSupportBufferError,
	NotSupportWeakRefError: () => NotSupportWeakRefError,
	Persistent: () => Persistent,
	RefTracker: () => RefTracker,
	Reference: () => Reference,
	ReferenceOwnership: () => ReferenceOwnership,
	ReferenceWithData: () => ReferenceWithData,
	ReferenceWithFinalizer: () => ReferenceWithFinalizer,
	ScopeStore: () => ScopeStore,
	Store: () => Store,
	TrackedFinalizer: () => TrackedFinalizer,
	TryCatch: () => TryCatch,
	createContext: () => createContext,
	getDefaultContext: () => getDefaultContext$1,
	getExternalValue: () => getExternalValue,
	isExternal: () => isExternal,
	isReferenceType: () => isReferenceType,
	version: () => version$1
});
/** @public */
function isExternal(object) {
	return externalValue.has(object);
}
/** @public */
function getExternalValue(external) {
	if (!isExternal(external)) throw new TypeError("not external");
	return externalValue.get(external);
}
function isReferenceType(v) {
	return typeof v === "object" && v !== null || typeof v === "function";
}
function throwNodeApiVersionError(moduleName, moduleApiVersion) {
	var errorMessage = "".concat(moduleName, " requires Node-API version ").concat(moduleApiVersion, ", but this version of Node.js only supports version ").concat(NODE_API_SUPPORTED_VERSION_MAX, " add-ons.");
	throw new Error(errorMessage);
}
function handleThrow(envObject, value) {
	if (envObject.terminatedOrTerminating()) return;
	throw value;
}
function newEnv(ctx, filename, moduleApiVersion, makeDynCall_vppp, makeDynCall_vp, abort, nodeBinding) {
	moduleApiVersion = typeof moduleApiVersion !== "number" ? NODE_API_DEFAULT_MODULE_API_VERSION : moduleApiVersion;
	if (moduleApiVersion < NODE_API_DEFAULT_MODULE_API_VERSION) moduleApiVersion = NODE_API_DEFAULT_MODULE_API_VERSION;
	else if (moduleApiVersion > NODE_API_SUPPORTED_VERSION_MAX && moduleApiVersion !== NAPI_VERSION_EXPERIMENTAL) throwNodeApiVersionError(filename, moduleApiVersion);
	var env = new NodeEnv(ctx, filename, moduleApiVersion, makeDynCall_vppp, makeDynCall_vp, abort, nodeBinding);
	ctx.envStore.add(env);
	ctx.addCleanupHook(env, function() {
		env.unref();
	}, 0);
	return env;
}
function canBeHeldWeakly(value) {
	return value.isObject() || value.isFunction() || value.isSymbol();
}
function createContext() {
	return new Context();
}
function getDefaultContext$1() {
	if (!defaultContext) defaultContext = createContext();
	return defaultContext;
}
var externalValue, External, supportNewFunction, _global, TryCatch, canSetFunctionName, supportReflect, supportFinalizer, supportWeakSymbol, supportBigInt, _require, _MessageChannel, _setImmediate, _Buffer, version$1, NODE_API_SUPPORTED_VERSION_MIN, NODE_API_SUPPORTED_VERSION_MAX, NAPI_VERSION_EXPERIMENTAL, NODE_API_DEFAULT_MODULE_API_VERSION, Handle, ConstHandle, HandleStore, HandleScope, ScopeStore, RefTracker, Finalizer, TrackedFinalizer, Env, NodeEnv, EmnapiError, NotSupportWeakRefError, NotSupportBufferError, StrongRef, Persistent, ReferenceOwnership, Reference, ReferenceWithData, ReferenceWithFinalizer, Deferred, Store, CleanupHookCallback, CleanupQueue, NodejsWaitingRequestCounter, Context, defaultContext;
var init_emnapi_esm_bundler = __esm({ "../../node_modules/.pnpm/@emnapi+runtime@1.5.0/node_modules/@emnapi/runtime/dist/emnapi.esm-bundler.js": (() => {
	init_tslib_es6();
	externalValue = /* @__PURE__ */ new WeakMap();
	External = (function() {
		function External$1(value) {
			Object.setPrototypeOf(this, null);
			externalValue.set(this, value);
		}
		External$1.prototype = null;
		return External$1;
	})();
	supportNewFunction = /* @__PURE__ */ (function() {
		var f;
		try {
			f = new Function();
		} catch (_) {
			return false;
		}
		return typeof f === "function";
	})();
	_global = /* @__PURE__ */ (function() {
		if (typeof globalThis !== "undefined") return globalThis;
		var g = (function() {
			return this;
		})();
		if (!g && supportNewFunction) try {
			g = new Function("return this")();
		} catch (_) {}
		if (!g) {
			if (typeof __webpack_public_path__ === "undefined") {
				if (typeof global !== "undefined") return global;
			}
			if (typeof window !== "undefined") return window;
			if (typeof self !== "undefined") return self;
		}
		return g;
	})();
	TryCatch = /* @__PURE__ */ function() {
		function TryCatch$1() {
			this._exception = void 0;
			this._caught = false;
		}
		TryCatch$1.prototype.isEmpty = function() {
			return !this._caught;
		};
		TryCatch$1.prototype.hasCaught = function() {
			return this._caught;
		};
		TryCatch$1.prototype.exception = function() {
			return this._exception;
		};
		TryCatch$1.prototype.setError = function(err) {
			this._caught = true;
			this._exception = err;
		};
		TryCatch$1.prototype.reset = function() {
			this._caught = false;
			this._exception = void 0;
		};
		TryCatch$1.prototype.extractException = function() {
			var e = this._exception;
			this.reset();
			return e;
		};
		return TryCatch$1;
	}();
	canSetFunctionName = /* @__PURE__ */ (function() {
		var _a;
		try {
			return Boolean((_a = Object.getOwnPropertyDescriptor(Function.prototype, "name")) === null || _a === void 0 ? void 0 : _a.configurable);
		} catch (_) {
			return false;
		}
	})();
	supportReflect = typeof Reflect === "object";
	supportFinalizer = typeof FinalizationRegistry !== "undefined" && typeof WeakRef !== "undefined";
	supportWeakSymbol = /* @__PURE__ */ (function() {
		try {
			var sym = Symbol();
			new WeakRef(sym);
			(/* @__PURE__ */ new WeakMap()).set(sym, void 0);
		} catch (_) {
			return false;
		}
		return true;
	})();
	supportBigInt = typeof BigInt !== "undefined";
	_require = /* @__PURE__ */ (function() {
		var nativeRequire;
		if (typeof __webpack_public_path__ !== "undefined") nativeRequire = (function() {
			return typeof __non_webpack_require__ !== "undefined" ? __non_webpack_require__ : void 0;
		})();
		else nativeRequire = (function() {
			return typeof __webpack_public_path__ !== "undefined" ? typeof __non_webpack_require__ !== "undefined" ? __non_webpack_require__ : void 0 : typeof __require !== "undefined" ? __require : void 0;
		})();
		return nativeRequire;
	})();
	_MessageChannel = typeof MessageChannel === "function" ? MessageChannel : /* @__PURE__ */ (function() {
		try {
			return _require("worker_threads").MessageChannel;
		} catch (_) {}
	})();
	_setImmediate = typeof setImmediate === "function" ? setImmediate : function(callback) {
		if (typeof callback !== "function") throw new TypeError("The \"callback\" argument must be of type function");
		if (_MessageChannel) {
			var channel_1 = new _MessageChannel();
			channel_1.port1.onmessage = function() {
				channel_1.port1.onmessage = null;
				channel_1 = void 0;
				callback();
			};
			channel_1.port2.postMessage(null);
		} else setTimeout(callback, 0);
	};
	_Buffer = typeof Buffer === "function" ? Buffer : /* @__PURE__ */ (function() {
		try {
			return _require("buffer").Buffer;
		} catch (_) {}
	})();
	version$1 = "1.5.0";
	NODE_API_SUPPORTED_VERSION_MIN = 1;
	NODE_API_SUPPORTED_VERSION_MAX = 10;
	NAPI_VERSION_EXPERIMENTAL = 2147483647;
	NODE_API_DEFAULT_MODULE_API_VERSION = 8;
	Handle = /* @__PURE__ */ function() {
		function Handle$1(id, value) {
			this.id = id;
			this.value = value;
		}
		Handle$1.prototype.data = function() {
			return getExternalValue(this.value);
		};
		Handle$1.prototype.isNumber = function() {
			return typeof this.value === "number";
		};
		Handle$1.prototype.isBigInt = function() {
			return typeof this.value === "bigint";
		};
		Handle$1.prototype.isString = function() {
			return typeof this.value === "string";
		};
		Handle$1.prototype.isFunction = function() {
			return typeof this.value === "function";
		};
		Handle$1.prototype.isExternal = function() {
			return isExternal(this.value);
		};
		Handle$1.prototype.isObject = function() {
			return typeof this.value === "object" && this.value !== null;
		};
		Handle$1.prototype.isArray = function() {
			return Array.isArray(this.value);
		};
		Handle$1.prototype.isArrayBuffer = function() {
			return this.value instanceof ArrayBuffer;
		};
		Handle$1.prototype.isTypedArray = function() {
			return ArrayBuffer.isView(this.value) && !(this.value instanceof DataView);
		};
		Handle$1.prototype.isBuffer = function(BufferConstructor) {
			if (ArrayBuffer.isView(this.value)) return true;
			BufferConstructor !== null && BufferConstructor !== void 0 || (BufferConstructor = _Buffer);
			return typeof BufferConstructor === "function" && BufferConstructor.isBuffer(this.value);
		};
		Handle$1.prototype.isDataView = function() {
			return this.value instanceof DataView;
		};
		Handle$1.prototype.isDate = function() {
			return this.value instanceof Date;
		};
		Handle$1.prototype.isPromise = function() {
			return this.value instanceof Promise;
		};
		Handle$1.prototype.isBoolean = function() {
			return typeof this.value === "boolean";
		};
		Handle$1.prototype.isUndefined = function() {
			return this.value === void 0;
		};
		Handle$1.prototype.isSymbol = function() {
			return typeof this.value === "symbol";
		};
		Handle$1.prototype.isNull = function() {
			return this.value === null;
		};
		Handle$1.prototype.dispose = function() {
			this.value = void 0;
		};
		return Handle$1;
	}();
	ConstHandle = /* @__PURE__ */ function(_super) {
		__extends(ConstHandle$1, _super);
		function ConstHandle$1(id, value) {
			return _super.call(this, id, value) || this;
		}
		ConstHandle$1.prototype.dispose = function() {};
		return ConstHandle$1;
	}(Handle);
	HandleStore = /* @__PURE__ */ function() {
		function HandleStore$1() {
			this._values = [
				void 0,
				HandleStore$1.UNDEFINED,
				HandleStore$1.NULL,
				HandleStore$1.FALSE,
				HandleStore$1.TRUE,
				HandleStore$1.GLOBAL
			];
			this._next = HandleStore$1.MIN_ID;
		}
		HandleStore$1.prototype.push = function(value) {
			var h;
			var next = this._next;
			var values = this._values;
			if (next < values.length) {
				h = values[next];
				h.value = value;
			} else {
				h = new Handle(next, value);
				values[next] = h;
			}
			this._next++;
			return h;
		};
		HandleStore$1.prototype.erase = function(start, end) {
			this._next = start;
			var values = this._values;
			for (var i = start; i < end; ++i) values[i].dispose();
		};
		HandleStore$1.prototype.get = function(id) {
			return this._values[id];
		};
		HandleStore$1.prototype.swap = function(a, b) {
			var values = this._values;
			var h = values[a];
			values[a] = values[b];
			values[a].id = Number(a);
			values[b] = h;
			h.id = Number(b);
		};
		HandleStore$1.prototype.dispose = function() {
			this._values.length = HandleStore$1.MIN_ID;
			this._next = HandleStore$1.MIN_ID;
		};
		HandleStore$1.UNDEFINED = new ConstHandle(1, void 0);
		HandleStore$1.NULL = new ConstHandle(2, null);
		HandleStore$1.FALSE = new ConstHandle(3, false);
		HandleStore$1.TRUE = new ConstHandle(4, true);
		HandleStore$1.GLOBAL = new ConstHandle(5, _global);
		HandleStore$1.MIN_ID = 6;
		return HandleStore$1;
	}();
	HandleScope = /* @__PURE__ */ function() {
		function HandleScope$1(handleStore, id, parentScope, start, end) {
			if (end === void 0) end = start;
			this.handleStore = handleStore;
			this.id = id;
			this.parent = parentScope;
			this.child = null;
			if (parentScope !== null) parentScope.child = this;
			this.start = start;
			this.end = end;
			this._escapeCalled = false;
			this.callbackInfo = {
				thiz: void 0,
				data: 0,
				args: void 0,
				fn: void 0
			};
		}
		HandleScope$1.prototype.add = function(value) {
			var h = this.handleStore.push(value);
			this.end++;
			return h;
		};
		HandleScope$1.prototype.addExternal = function(data) {
			return this.add(new External(data));
		};
		HandleScope$1.prototype.dispose = function() {
			if (this._escapeCalled) this._escapeCalled = false;
			if (this.start === this.end) return;
			this.handleStore.erase(this.start, this.end);
		};
		HandleScope$1.prototype.escape = function(handle) {
			if (this._escapeCalled) return null;
			this._escapeCalled = true;
			if (handle < this.start || handle >= this.end) return null;
			this.handleStore.swap(handle, this.start);
			var h = this.handleStore.get(this.start);
			this.start++;
			this.parent.end++;
			return h;
		};
		HandleScope$1.prototype.escapeCalled = function() {
			return this._escapeCalled;
		};
		return HandleScope$1;
	}();
	ScopeStore = /* @__PURE__ */ function() {
		function ScopeStore$1() {
			this._rootScope = new HandleScope(null, 0, null, 1, HandleStore.MIN_ID);
			this.currentScope = this._rootScope;
			this._values = [void 0];
		}
		ScopeStore$1.prototype.get = function(id) {
			return this._values[id];
		};
		ScopeStore$1.prototype.openScope = function(handleStore) {
			var currentScope = this.currentScope;
			var scope = currentScope.child;
			if (scope !== null) scope.start = scope.end = currentScope.end;
			else {
				var id = currentScope.id + 1;
				scope = new HandleScope(handleStore, id, currentScope, currentScope.end);
				this._values[id] = scope;
			}
			this.currentScope = scope;
			return scope;
		};
		ScopeStore$1.prototype.closeScope = function() {
			var scope = this.currentScope;
			this.currentScope = scope.parent;
			scope.dispose();
		};
		ScopeStore$1.prototype.dispose = function() {
			this.currentScope = this._rootScope;
			this._values.length = 1;
		};
		return ScopeStore$1;
	}();
	RefTracker = /* @__PURE__ */ function() {
		function RefTracker$1() {
			this._next = null;
			this._prev = null;
		}
		/** @virtual */
		RefTracker$1.prototype.dispose = function() {};
		/** @virtual */
		RefTracker$1.prototype.finalize = function() {};
		RefTracker$1.prototype.link = function(list) {
			this._prev = list;
			this._next = list._next;
			if (this._next !== null) this._next._prev = this;
			list._next = this;
		};
		RefTracker$1.prototype.unlink = function() {
			if (this._prev !== null) this._prev._next = this._next;
			if (this._next !== null) this._next._prev = this._prev;
			this._prev = null;
			this._next = null;
		};
		RefTracker$1.finalizeAll = function(list) {
			while (list._next !== null) list._next.finalize();
		};
		return RefTracker$1;
	}();
	Finalizer = /* @__PURE__ */ function() {
		function Finalizer$1(envObject, _finalizeCallback, _finalizeData, _finalizeHint) {
			if (_finalizeCallback === void 0) _finalizeCallback = 0;
			if (_finalizeData === void 0) _finalizeData = 0;
			if (_finalizeHint === void 0) _finalizeHint = 0;
			this.envObject = envObject;
			this._finalizeCallback = _finalizeCallback;
			this._finalizeData = _finalizeData;
			this._finalizeHint = _finalizeHint;
			this._makeDynCall_vppp = envObject.makeDynCall_vppp;
		}
		Finalizer$1.prototype.callback = function() {
			return this._finalizeCallback;
		};
		Finalizer$1.prototype.data = function() {
			return this._finalizeData;
		};
		Finalizer$1.prototype.hint = function() {
			return this._finalizeHint;
		};
		Finalizer$1.prototype.resetEnv = function() {
			this.envObject = void 0;
		};
		Finalizer$1.prototype.resetFinalizer = function() {
			this._finalizeCallback = 0;
			this._finalizeData = 0;
			this._finalizeHint = 0;
		};
		Finalizer$1.prototype.callFinalizer = function() {
			var finalize_callback = this._finalizeCallback;
			var finalize_data = this._finalizeData;
			var finalize_hint = this._finalizeHint;
			this.resetFinalizer();
			if (!finalize_callback) return;
			var fini = Number(finalize_callback);
			if (!this.envObject) this._makeDynCall_vppp(fini)(0, finalize_data, finalize_hint);
			else this.envObject.callFinalizer(fini, finalize_data, finalize_hint);
		};
		Finalizer$1.prototype.dispose = function() {
			this.envObject = void 0;
			this._makeDynCall_vppp = void 0;
		};
		return Finalizer$1;
	}();
	TrackedFinalizer = /* @__PURE__ */ function(_super) {
		__extends(TrackedFinalizer$1, _super);
		function TrackedFinalizer$1(envObject, finalize_callback, finalize_data, finalize_hint) {
			var _this = _super.call(this) || this;
			_this._finalizer = new Finalizer(envObject, finalize_callback, finalize_data, finalize_hint);
			return _this;
		}
		TrackedFinalizer$1.create = function(envObject, finalize_callback, finalize_data, finalize_hint) {
			var finalizer = new TrackedFinalizer$1(envObject, finalize_callback, finalize_data, finalize_hint);
			finalizer.link(envObject.finalizing_reflist);
			return finalizer;
		};
		TrackedFinalizer$1.prototype.data = function() {
			return this._finalizer.data();
		};
		TrackedFinalizer$1.prototype.dispose = function() {
			if (!this._finalizer) return;
			this.unlink();
			this._finalizer.envObject.dequeueFinalizer(this);
			this._finalizer.dispose();
			this._finalizer = void 0;
			_super.prototype.dispose.call(this);
		};
		TrackedFinalizer$1.prototype.finalize = function() {
			this.unlink();
			var error$1;
			var caught = false;
			try {
				this._finalizer.callFinalizer();
			} catch (err) {
				caught = true;
				error$1 = err;
			}
			this.dispose();
			if (caught) throw error$1;
		};
		return TrackedFinalizer$1;
	}(RefTracker);
	Env = /* @__PURE__ */ function() {
		function Env$1(ctx, moduleApiVersion, makeDynCall_vppp, makeDynCall_vp, abort) {
			this.ctx = ctx;
			this.moduleApiVersion = moduleApiVersion;
			this.makeDynCall_vppp = makeDynCall_vppp;
			this.makeDynCall_vp = makeDynCall_vp;
			this.abort = abort;
			this.openHandleScopes = 0;
			this.instanceData = null;
			this.tryCatch = new TryCatch();
			this.refs = 1;
			this.reflist = new RefTracker();
			this.finalizing_reflist = new RefTracker();
			this.pendingFinalizers = [];
			this.lastError = {
				errorCode: 0,
				engineErrorCode: 0,
				engineReserved: 0
			};
			this.inGcFinalizer = false;
			this._bindingMap = /* @__PURE__ */ new WeakMap();
			this.id = 0;
		}
		/** @virtual */
		Env$1.prototype.canCallIntoJs = function() {
			return true;
		};
		Env$1.prototype.terminatedOrTerminating = function() {
			return !this.canCallIntoJs();
		};
		Env$1.prototype.ref = function() {
			this.refs++;
		};
		Env$1.prototype.unref = function() {
			this.refs--;
			if (this.refs === 0) this.dispose();
		};
		Env$1.prototype.ensureHandle = function(value) {
			return this.ctx.ensureHandle(value);
		};
		Env$1.prototype.ensureHandleId = function(value) {
			return this.ensureHandle(value).id;
		};
		Env$1.prototype.clearLastError = function() {
			var lastError = this.lastError;
			if (lastError.errorCode !== 0) lastError.errorCode = 0;
			if (lastError.engineErrorCode !== 0) lastError.engineErrorCode = 0;
			if (lastError.engineReserved !== 0) lastError.engineReserved = 0;
			return 0;
		};
		Env$1.prototype.setLastError = function(error_code, engine_error_code, engine_reserved) {
			if (engine_error_code === void 0) engine_error_code = 0;
			if (engine_reserved === void 0) engine_reserved = 0;
			var lastError = this.lastError;
			if (lastError.errorCode !== error_code) lastError.errorCode = error_code;
			if (lastError.engineErrorCode !== engine_error_code) lastError.engineErrorCode = engine_error_code;
			if (lastError.engineReserved !== engine_reserved) lastError.engineReserved = engine_reserved;
			return error_code;
		};
		Env$1.prototype.getReturnStatus = function() {
			return !this.tryCatch.hasCaught() ? 0 : this.setLastError(10);
		};
		Env$1.prototype.callIntoModule = function(fn, handleException) {
			if (handleException === void 0) handleException = handleThrow;
			var openHandleScopesBefore = this.openHandleScopes;
			this.clearLastError();
			var r = fn(this);
			if (openHandleScopesBefore !== this.openHandleScopes) this.abort("open_handle_scopes != open_handle_scopes_before");
			if (this.tryCatch.hasCaught()) {
				var err = this.tryCatch.extractException();
				handleException(this, err);
			}
			return r;
		};
		Env$1.prototype.invokeFinalizerFromGC = function(finalizer) {
			if (this.moduleApiVersion !== NAPI_VERSION_EXPERIMENTAL) this.enqueueFinalizer(finalizer);
			else {
				var saved = this.inGcFinalizer;
				this.inGcFinalizer = true;
				try {
					finalizer.finalize();
				} finally {
					this.inGcFinalizer = saved;
				}
			}
		};
		Env$1.prototype.checkGCAccess = function() {
			if (this.moduleApiVersion === NAPI_VERSION_EXPERIMENTAL && this.inGcFinalizer) this.abort("Finalizer is calling a function that may affect GC state.\nThe finalizers are run directly from GC and must not affect GC state.\nUse `node_api_post_finalizer` from inside of the finalizer to work around this issue.\nIt schedules the call as a new task in the event loop.");
		};
		/** @virtual */
		Env$1.prototype.enqueueFinalizer = function(finalizer) {
			if (this.pendingFinalizers.indexOf(finalizer) === -1) this.pendingFinalizers.push(finalizer);
		};
		/** @virtual */
		Env$1.prototype.dequeueFinalizer = function(finalizer) {
			var index = this.pendingFinalizers.indexOf(finalizer);
			if (index !== -1) this.pendingFinalizers.splice(index, 1);
		};
		/** @virtual */
		Env$1.prototype.deleteMe = function() {
			RefTracker.finalizeAll(this.finalizing_reflist);
			RefTracker.finalizeAll(this.reflist);
			this.tryCatch.extractException();
			this.ctx.envStore.remove(this.id);
		};
		Env$1.prototype.dispose = function() {
			if (this.id === 0) return;
			this.deleteMe();
			this.finalizing_reflist.dispose();
			this.reflist.dispose();
			this.id = 0;
		};
		Env$1.prototype.initObjectBinding = function(value) {
			var binding = {
				wrapped: 0,
				tag: null
			};
			this._bindingMap.set(value, binding);
			return binding;
		};
		Env$1.prototype.getObjectBinding = function(value) {
			if (this._bindingMap.has(value)) return this._bindingMap.get(value);
			return this.initObjectBinding(value);
		};
		Env$1.prototype.setInstanceData = function(data, finalize_cb, finalize_hint) {
			if (this.instanceData) this.instanceData.dispose();
			this.instanceData = TrackedFinalizer.create(this, finalize_cb, data, finalize_hint);
		};
		Env$1.prototype.getInstanceData = function() {
			return this.instanceData ? this.instanceData.data() : 0;
		};
		return Env$1;
	}();
	NodeEnv = /* @__PURE__ */ function(_super) {
		__extends(NodeEnv$1, _super);
		function NodeEnv$1(ctx, filename, moduleApiVersion, makeDynCall_vppp, makeDynCall_vp, abort, nodeBinding) {
			var _this = _super.call(this, ctx, moduleApiVersion, makeDynCall_vppp, makeDynCall_vp, abort) || this;
			_this.filename = filename;
			_this.nodeBinding = nodeBinding;
			_this.destructing = false;
			_this.finalizationScheduled = false;
			return _this;
		}
		NodeEnv$1.prototype.deleteMe = function() {
			this.destructing = true;
			this.drainFinalizerQueue();
			_super.prototype.deleteMe.call(this);
		};
		NodeEnv$1.prototype.canCallIntoJs = function() {
			return _super.prototype.canCallIntoJs.call(this) && this.ctx.canCallIntoJs();
		};
		NodeEnv$1.prototype.triggerFatalException = function(err) {
			if (this.nodeBinding) this.nodeBinding.napi.fatalException(err);
			else if (typeof process === "object" && process !== null && typeof process._fatalException === "function") {
				if (!process._fatalException(err)) {
					console.error(err);
					process.exit(1);
				}
			} else throw err;
		};
		NodeEnv$1.prototype.callbackIntoModule = function(enforceUncaughtExceptionPolicy, fn) {
			return this.callIntoModule(fn, function(envObject, err) {
				if (envObject.terminatedOrTerminating()) return;
				var hasProcess = typeof process === "object" && process !== null;
				var hasForceFlag = hasProcess ? Boolean(process.execArgv && process.execArgv.indexOf("--force-node-api-uncaught-exceptions-policy") !== -1) : false;
				if (envObject.moduleApiVersion < 10 && !hasForceFlag && !enforceUncaughtExceptionPolicy) {
					(hasProcess && typeof process.emitWarning === "function" ? process.emitWarning : function(warning, type, code) {
						if (warning instanceof Error) console.warn(warning.toString());
						else {
							var prefix = code ? "[".concat(code, "] ") : "";
							console.warn("".concat(prefix).concat(type || "Warning", ": ").concat(warning));
						}
					})("Uncaught N-API callback exception detected, please run node with option --force-node-api-uncaught-exceptions-policy=true to handle those exceptions properly.", "DeprecationWarning", "DEP0168");
					return;
				}
				envObject.triggerFatalException(err);
			});
		};
		NodeEnv$1.prototype.callFinalizer = function(cb, data, hint) {
			this.callFinalizerInternal(1, cb, data, hint);
		};
		NodeEnv$1.prototype.callFinalizerInternal = function(forceUncaught, cb, data, hint) {
			var f = this.makeDynCall_vppp(cb);
			var env = this.id;
			var scope = this.ctx.openScope(this);
			try {
				this.callbackIntoModule(Boolean(forceUncaught), function() {
					f(env, data, hint);
				});
			} finally {
				this.ctx.closeScope(this, scope);
			}
		};
		NodeEnv$1.prototype.enqueueFinalizer = function(finalizer) {
			var _this = this;
			_super.prototype.enqueueFinalizer.call(this, finalizer);
			if (!this.finalizationScheduled && !this.destructing) {
				this.finalizationScheduled = true;
				this.ref();
				_setImmediate(function() {
					_this.finalizationScheduled = false;
					_this.unref();
					_this.drainFinalizerQueue();
				});
			}
		};
		NodeEnv$1.prototype.drainFinalizerQueue = function() {
			while (this.pendingFinalizers.length > 0) this.pendingFinalizers.shift().finalize();
		};
		return NodeEnv$1;
	}(Env);
	EmnapiError = /* @__PURE__ */ function(_super) {
		__extends(EmnapiError$1, _super);
		function EmnapiError$1(message) {
			var _newTarget = this.constructor;
			var _this = _super.call(this, message) || this;
			var ErrorConstructor = _newTarget;
			var proto = ErrorConstructor.prototype;
			if (!(_this instanceof EmnapiError$1)) {
				var setPrototypeOf = Object.setPrototypeOf;
				if (typeof setPrototypeOf === "function") setPrototypeOf.call(Object, _this, proto);
				else _this.__proto__ = proto;
				if (typeof Error.captureStackTrace === "function") Error.captureStackTrace(_this, ErrorConstructor);
			}
			return _this;
		}
		return EmnapiError$1;
	}(Error);
	Object.defineProperty(EmnapiError.prototype, "name", {
		configurable: true,
		writable: true,
		value: "EmnapiError"
	});
	NotSupportWeakRefError = /* @__PURE__ */ function(_super) {
		__extends(NotSupportWeakRefError$1, _super);
		function NotSupportWeakRefError$1(api, message) {
			return _super.call(this, "".concat(api, ": The current runtime does not support \"FinalizationRegistry\" and \"WeakRef\".").concat(message ? " ".concat(message) : "")) || this;
		}
		return NotSupportWeakRefError$1;
	}(EmnapiError);
	Object.defineProperty(NotSupportWeakRefError.prototype, "name", {
		configurable: true,
		writable: true,
		value: "NotSupportWeakRefError"
	});
	NotSupportBufferError = /* @__PURE__ */ function(_super) {
		__extends(NotSupportBufferError$1, _super);
		function NotSupportBufferError$1(api, message) {
			return _super.call(this, "".concat(api, ": The current runtime does not support \"Buffer\". Consider using buffer polyfill to make sure `globalThis.Buffer` is defined.").concat(message ? " ".concat(message) : "")) || this;
		}
		return NotSupportBufferError$1;
	}(EmnapiError);
	Object.defineProperty(NotSupportBufferError.prototype, "name", {
		configurable: true,
		writable: true,
		value: "NotSupportBufferError"
	});
	StrongRef = /* @__PURE__ */ function() {
		function StrongRef$1(value) {
			this._value = value;
		}
		StrongRef$1.prototype.deref = function() {
			return this._value;
		};
		StrongRef$1.prototype.dispose = function() {
			this._value = void 0;
		};
		return StrongRef$1;
	}();
	Persistent = /* @__PURE__ */ function() {
		function Persistent$1(value) {
			this._ref = new StrongRef(value);
		}
		Persistent$1.prototype.setWeak = function(param, callback) {
			if (!supportFinalizer || this._ref === void 0 || this._ref instanceof WeakRef) return;
			var value = this._ref.deref();
			try {
				Persistent$1._registry.register(value, this, this);
				var weakRef = new WeakRef(value);
				this._ref.dispose();
				this._ref = weakRef;
				this._param = param;
				this._callback = callback;
			} catch (err) {
				if (typeof value === "symbol");
				else throw err;
			}
		};
		Persistent$1.prototype.clearWeak = function() {
			if (!supportFinalizer || this._ref === void 0) return;
			if (this._ref instanceof WeakRef) {
				try {
					Persistent$1._registry.unregister(this);
				} catch (_) {}
				this._param = void 0;
				this._callback = void 0;
				var value = this._ref.deref();
				if (value === void 0) this._ref = value;
				else this._ref = new StrongRef(value);
			}
		};
		Persistent$1.prototype.reset = function() {
			if (supportFinalizer) try {
				Persistent$1._registry.unregister(this);
			} catch (_) {}
			this._param = void 0;
			this._callback = void 0;
			if (this._ref instanceof StrongRef) this._ref.dispose();
			this._ref = void 0;
		};
		Persistent$1.prototype.isEmpty = function() {
			return this._ref === void 0;
		};
		Persistent$1.prototype.deref = function() {
			if (this._ref === void 0) return void 0;
			return this._ref.deref();
		};
		Persistent$1._registry = supportFinalizer ? new FinalizationRegistry(function(value) {
			value._ref = void 0;
			var callback = value._callback;
			var param = value._param;
			value._callback = void 0;
			value._param = void 0;
			if (typeof callback === "function") callback(param);
		}) : void 0;
		return Persistent$1;
	}();
	;
	(function(ReferenceOwnership$1) {
		ReferenceOwnership$1[ReferenceOwnership$1["kRuntime"] = 0] = "kRuntime";
		ReferenceOwnership$1[ReferenceOwnership$1["kUserland"] = 1] = "kUserland";
	})(ReferenceOwnership || (ReferenceOwnership = {}));
	Reference = /* @__PURE__ */ function(_super) {
		__extends(Reference$1, _super);
		function Reference$1(envObject, handle_id, initialRefcount, ownership) {
			var _this = _super.call(this) || this;
			_this.envObject = envObject;
			_this._refcount = initialRefcount;
			_this._ownership = ownership;
			var handle = envObject.ctx.handleStore.get(handle_id);
			_this.canBeWeak = canBeHeldWeakly(handle);
			_this.persistent = new Persistent(handle.value);
			_this.id = 0;
			if (initialRefcount === 0) _this._setWeak();
			return _this;
		}
		Reference$1.weakCallback = function(ref) {
			ref.persistent.reset();
			ref.invokeFinalizerFromGC();
		};
		Reference$1.create = function(envObject, handle_id, initialRefcount, ownership, _unused1, _unused2, _unused3) {
			var ref = new Reference$1(envObject, handle_id, initialRefcount, ownership);
			envObject.ctx.refStore.add(ref);
			ref.link(envObject.reflist);
			return ref;
		};
		Reference$1.prototype.ref = function() {
			if (this.persistent.isEmpty()) return 0;
			if (++this._refcount === 1 && this.canBeWeak) this.persistent.clearWeak();
			return this._refcount;
		};
		Reference$1.prototype.unref = function() {
			if (this.persistent.isEmpty() || this._refcount === 0) return 0;
			if (--this._refcount === 0) this._setWeak();
			return this._refcount;
		};
		Reference$1.prototype.get = function(envObject) {
			if (envObject === void 0) envObject = this.envObject;
			if (this.persistent.isEmpty()) return 0;
			var obj = this.persistent.deref();
			return envObject.ensureHandle(obj).id;
		};
		/** @virtual */
		Reference$1.prototype.resetFinalizer = function() {};
		/** @virtual */
		Reference$1.prototype.data = function() {
			return 0;
		};
		Reference$1.prototype.refcount = function() {
			return this._refcount;
		};
		Reference$1.prototype.ownership = function() {
			return this._ownership;
		};
		/** @virtual */
		Reference$1.prototype.callUserFinalizer = function() {};
		/** @virtual */
		Reference$1.prototype.invokeFinalizerFromGC = function() {
			this.finalize();
		};
		Reference$1.prototype._setWeak = function() {
			if (this.canBeWeak) this.persistent.setWeak(this, Reference$1.weakCallback);
			else this.persistent.reset();
		};
		Reference$1.prototype.finalize = function() {
			this.persistent.reset();
			var deleteMe = this._ownership === ReferenceOwnership.kRuntime;
			this.unlink();
			this.callUserFinalizer();
			if (deleteMe) this.dispose();
		};
		Reference$1.prototype.dispose = function() {
			if (this.id === 0) return;
			this.unlink();
			this.persistent.reset();
			this.envObject.ctx.refStore.remove(this.id);
			_super.prototype.dispose.call(this);
			this.envObject = void 0;
			this.id = 0;
		};
		return Reference$1;
	}(RefTracker);
	ReferenceWithData = /* @__PURE__ */ function(_super) {
		__extends(ReferenceWithData$1, _super);
		function ReferenceWithData$1(envObject, value, initialRefcount, ownership, _data) {
			var _this = _super.call(this, envObject, value, initialRefcount, ownership) || this;
			_this._data = _data;
			return _this;
		}
		ReferenceWithData$1.create = function(envObject, value, initialRefcount, ownership, data) {
			var reference = new ReferenceWithData$1(envObject, value, initialRefcount, ownership, data);
			envObject.ctx.refStore.add(reference);
			reference.link(envObject.reflist);
			return reference;
		};
		ReferenceWithData$1.prototype.data = function() {
			return this._data;
		};
		return ReferenceWithData$1;
	}(Reference);
	ReferenceWithFinalizer = /* @__PURE__ */ function(_super) {
		__extends(ReferenceWithFinalizer$1, _super);
		function ReferenceWithFinalizer$1(envObject, value, initialRefcount, ownership, finalize_callback, finalize_data, finalize_hint) {
			var _this = _super.call(this, envObject, value, initialRefcount, ownership) || this;
			_this._finalizer = new Finalizer(envObject, finalize_callback, finalize_data, finalize_hint);
			return _this;
		}
		ReferenceWithFinalizer$1.create = function(envObject, value, initialRefcount, ownership, finalize_callback, finalize_data, finalize_hint) {
			var reference = new ReferenceWithFinalizer$1(envObject, value, initialRefcount, ownership, finalize_callback, finalize_data, finalize_hint);
			envObject.ctx.refStore.add(reference);
			reference.link(envObject.finalizing_reflist);
			return reference;
		};
		ReferenceWithFinalizer$1.prototype.resetFinalizer = function() {
			this._finalizer.resetFinalizer();
		};
		ReferenceWithFinalizer$1.prototype.data = function() {
			return this._finalizer.data();
		};
		ReferenceWithFinalizer$1.prototype.callUserFinalizer = function() {
			this._finalizer.callFinalizer();
		};
		ReferenceWithFinalizer$1.prototype.invokeFinalizerFromGC = function() {
			this._finalizer.envObject.invokeFinalizerFromGC(this);
		};
		ReferenceWithFinalizer$1.prototype.dispose = function() {
			if (!this._finalizer) return;
			this._finalizer.envObject.dequeueFinalizer(this);
			this._finalizer.dispose();
			_super.prototype.dispose.call(this);
			this._finalizer = void 0;
		};
		return ReferenceWithFinalizer$1;
	}(Reference);
	Deferred = /* @__PURE__ */ function() {
		function Deferred$1(ctx, value) {
			this.id = 0;
			this.ctx = ctx;
			this.value = value;
		}
		Deferred$1.create = function(ctx, value) {
			var deferred = new Deferred$1(ctx, value);
			ctx.deferredStore.add(deferred);
			return deferred;
		};
		Deferred$1.prototype.resolve = function(value) {
			this.value.resolve(value);
			this.dispose();
		};
		Deferred$1.prototype.reject = function(reason) {
			this.value.reject(reason);
			this.dispose();
		};
		Deferred$1.prototype.dispose = function() {
			this.ctx.deferredStore.remove(this.id);
			this.id = 0;
			this.value = null;
			this.ctx = null;
		};
		return Deferred$1;
	}();
	Store = /* @__PURE__ */ function() {
		function Store$1() {
			this._values = [void 0];
			this._values.length = 4;
			this._size = 1;
			this._freeList = [];
		}
		Store$1.prototype.add = function(value) {
			var id;
			if (this._freeList.length) id = this._freeList.shift();
			else {
				id = this._size;
				this._size++;
				var capacity = this._values.length;
				if (id >= capacity) this._values.length = capacity + (capacity >> 1) + 16;
			}
			value.id = id;
			this._values[id] = value;
		};
		Store$1.prototype.get = function(id) {
			return this._values[id];
		};
		Store$1.prototype.has = function(id) {
			return this._values[id] !== void 0;
		};
		Store$1.prototype.remove = function(id) {
			var value = this._values[id];
			if (value) {
				value.id = 0;
				this._values[id] = void 0;
				this._freeList.push(Number(id));
			}
		};
		Store$1.prototype.dispose = function() {
			for (var i = 1; i < this._size; ++i) {
				var value = this._values[i];
				value === null || value === void 0 || value.dispose();
			}
			this._values = [void 0];
			this._size = 1;
			this._freeList = [];
		};
		return Store$1;
	}();
	CleanupHookCallback = /* @__PURE__ */ function() {
		function CleanupHookCallback$1(envObject, fn, arg, order) {
			this.envObject = envObject;
			this.fn = fn;
			this.arg = arg;
			this.order = order;
		}
		return CleanupHookCallback$1;
	}();
	CleanupQueue = /* @__PURE__ */ function() {
		function CleanupQueue$1() {
			this._cleanupHooks = [];
			this._cleanupHookCounter = 0;
		}
		CleanupQueue$1.prototype.empty = function() {
			return this._cleanupHooks.length === 0;
		};
		CleanupQueue$1.prototype.add = function(envObject, fn, arg) {
			if (this._cleanupHooks.filter(function(hook) {
				return hook.envObject === envObject && hook.fn === fn && hook.arg === arg;
			}).length > 0) throw new Error("Can not add same fn and arg twice");
			this._cleanupHooks.push(new CleanupHookCallback(envObject, fn, arg, this._cleanupHookCounter++));
		};
		CleanupQueue$1.prototype.remove = function(envObject, fn, arg) {
			for (var i = 0; i < this._cleanupHooks.length; ++i) {
				var hook = this._cleanupHooks[i];
				if (hook.envObject === envObject && hook.fn === fn && hook.arg === arg) {
					this._cleanupHooks.splice(i, 1);
					return;
				}
			}
		};
		CleanupQueue$1.prototype.drain = function() {
			var hooks = this._cleanupHooks.slice();
			hooks.sort(function(a, b) {
				return b.order - a.order;
			});
			for (var i = 0; i < hooks.length; ++i) {
				var cb = hooks[i];
				if (typeof cb.fn === "number") cb.envObject.makeDynCall_vp(cb.fn)(cb.arg);
				else cb.fn(cb.arg);
				this._cleanupHooks.splice(this._cleanupHooks.indexOf(cb), 1);
			}
		};
		CleanupQueue$1.prototype.dispose = function() {
			this._cleanupHooks.length = 0;
			this._cleanupHookCounter = 0;
		};
		return CleanupQueue$1;
	}();
	NodejsWaitingRequestCounter = /* @__PURE__ */ function() {
		function NodejsWaitingRequestCounter$1() {
			this.refHandle = new _MessageChannel().port1;
			this.count = 0;
		}
		NodejsWaitingRequestCounter$1.prototype.increase = function() {
			if (this.count === 0) {
				if (this.refHandle.ref) this.refHandle.ref();
			}
			this.count++;
		};
		NodejsWaitingRequestCounter$1.prototype.decrease = function() {
			if (this.count === 0) return;
			if (this.count === 1) {
				if (this.refHandle.unref) this.refHandle.unref();
			}
			this.count--;
		};
		return NodejsWaitingRequestCounter$1;
	}();
	Context = /* @__PURE__ */ function() {
		function Context$1() {
			var _this = this;
			this._isStopping = false;
			this._canCallIntoJs = true;
			this._suppressDestroy = false;
			this.envStore = new Store();
			this.scopeStore = new ScopeStore();
			this.refStore = new Store();
			this.deferredStore = new Store();
			this.handleStore = new HandleStore();
			this.feature = {
				supportReflect,
				supportFinalizer,
				supportWeakSymbol,
				supportBigInt,
				supportNewFunction,
				canSetFunctionName,
				setImmediate: _setImmediate,
				Buffer: _Buffer,
				MessageChannel: _MessageChannel
			};
			this.cleanupQueue = new CleanupQueue();
			if (typeof process === "object" && process !== null && typeof process.once === "function") {
				this.refCounter = new NodejsWaitingRequestCounter();
				process.once("beforeExit", function() {
					if (!_this._suppressDestroy) _this.destroy();
				});
			}
		}
		/**
		* Suppress the destroy on `beforeExit` event in Node.js.
		* Call this method if you want to keep the context and
		* all associated {@link Env | Env} alive,
		* this also means that cleanup hooks will not be called.
		* After call this method, you should call
		* {@link Context.destroy | `Context.prototype.destroy`} method manually.
		*/
		Context$1.prototype.suppressDestroy = function() {
			this._suppressDestroy = true;
		};
		Context$1.prototype.getRuntimeVersions = function() {
			return {
				version: version$1,
				NODE_API_SUPPORTED_VERSION_MAX,
				NAPI_VERSION_EXPERIMENTAL,
				NODE_API_DEFAULT_MODULE_API_VERSION
			};
		};
		Context$1.prototype.createNotSupportWeakRefError = function(api, message) {
			return new NotSupportWeakRefError(api, message);
		};
		Context$1.prototype.createNotSupportBufferError = function(api, message) {
			return new NotSupportBufferError(api, message);
		};
		Context$1.prototype.createReference = function(envObject, handle_id, initialRefcount, ownership) {
			return Reference.create(envObject, handle_id, initialRefcount, ownership);
		};
		Context$1.prototype.createReferenceWithData = function(envObject, handle_id, initialRefcount, ownership, data) {
			return ReferenceWithData.create(envObject, handle_id, initialRefcount, ownership, data);
		};
		Context$1.prototype.createReferenceWithFinalizer = function(envObject, handle_id, initialRefcount, ownership, finalize_callback, finalize_data, finalize_hint) {
			if (finalize_callback === void 0) finalize_callback = 0;
			if (finalize_data === void 0) finalize_data = 0;
			if (finalize_hint === void 0) finalize_hint = 0;
			return ReferenceWithFinalizer.create(envObject, handle_id, initialRefcount, ownership, finalize_callback, finalize_data, finalize_hint);
		};
		Context$1.prototype.createDeferred = function(value) {
			return Deferred.create(this, value);
		};
		Context$1.prototype.createEnv = function(filename, moduleApiVersion, makeDynCall_vppp, makeDynCall_vp, abort, nodeBinding) {
			return newEnv(this, filename, moduleApiVersion, makeDynCall_vppp, makeDynCall_vp, abort, nodeBinding);
		};
		Context$1.prototype.createTrackedFinalizer = function(envObject, finalize_callback, finalize_data, finalize_hint) {
			return TrackedFinalizer.create(envObject, finalize_callback, finalize_data, finalize_hint);
		};
		Context$1.prototype.getCurrentScope = function() {
			return this.scopeStore.currentScope;
		};
		Context$1.prototype.addToCurrentScope = function(value) {
			return this.scopeStore.currentScope.add(value);
		};
		Context$1.prototype.openScope = function(envObject) {
			var scope = this.scopeStore.openScope(this.handleStore);
			if (envObject) envObject.openHandleScopes++;
			return scope;
		};
		Context$1.prototype.closeScope = function(envObject, _scope) {
			if (envObject && envObject.openHandleScopes === 0) return;
			this.scopeStore.closeScope();
			if (envObject) envObject.openHandleScopes--;
		};
		Context$1.prototype.ensureHandle = function(value) {
			switch (value) {
				case void 0: return HandleStore.UNDEFINED;
				case null: return HandleStore.NULL;
				case true: return HandleStore.TRUE;
				case false: return HandleStore.FALSE;
				case _global: return HandleStore.GLOBAL;
			}
			return this.addToCurrentScope(value);
		};
		Context$1.prototype.addCleanupHook = function(envObject, fn, arg) {
			this.cleanupQueue.add(envObject, fn, arg);
		};
		Context$1.prototype.removeCleanupHook = function(envObject, fn, arg) {
			this.cleanupQueue.remove(envObject, fn, arg);
		};
		Context$1.prototype.runCleanup = function() {
			while (!this.cleanupQueue.empty()) this.cleanupQueue.drain();
		};
		Context$1.prototype.increaseWaitingRequestCounter = function() {
			var _a;
			(_a = this.refCounter) === null || _a === void 0 || _a.increase();
		};
		Context$1.prototype.decreaseWaitingRequestCounter = function() {
			var _a;
			(_a = this.refCounter) === null || _a === void 0 || _a.decrease();
		};
		Context$1.prototype.setCanCallIntoJs = function(value) {
			this._canCallIntoJs = value;
		};
		Context$1.prototype.setStopping = function(value) {
			this._isStopping = value;
		};
		Context$1.prototype.canCallIntoJs = function() {
			return this._canCallIntoJs && !this._isStopping;
		};
		/**
		* Destroy the context and call cleanup hooks.
		* Associated {@link Env | Env} will be destroyed.
		*/
		Context$1.prototype.destroy = function() {
			this.setStopping(true);
			this.setCanCallIntoJs(false);
			this.runCleanup();
		};
		return Context$1;
	}();
	;
}) });

//#endregion
//#region ../../node_modules/.pnpm/@tybys+wasm-util@0.10.1/node_modules/@tybys/wasm-util/dist/wasm-util.esm-bundler.js
var wasm_util_esm_bundler_exports = /* @__PURE__ */ __export({
	Asyncify: () => Asyncify,
	Memory: () => Memory,
	WASI: () => WASI$2,
	WebAssemblyMemory: () => WebAssemblyMemory,
	asyncifyLoad: () => asyncifyLoad,
	asyncifyLoadSync: () => asyncifyLoadSync,
	createAsyncWASI: () => createAsyncWASI,
	extendMemory: () => extendMemory,
	load: () => load,
	loadSync: () => loadSync,
	wrapAsyncExport: () => wrapAsyncExport,
	wrapAsyncImport: () => wrapAsyncImport,
	wrapExports: () => wrapExports
});
function validateObject(value, name) {
	if (value === null || typeof value !== "object") throw new TypeError(`${name} must be an object. Received ${value === null ? "null" : typeof value}`);
}
function validateArray(value, name) {
	if (!Array.isArray(value)) throw new TypeError(`${name} must be an array. Received ${value === null ? "null" : typeof value}`);
}
function validateBoolean(value, name) {
	if (typeof value !== "boolean") throw new TypeError(`${name} must be a boolean. Received ${value === null ? "null" : typeof value}`);
}
function validateString(value, name) {
	if (typeof value !== "string") throw new TypeError(`${name} must be a string. Received ${value === null ? "null" : typeof value}`);
}
function validateFunction(value, name) {
	if (typeof value !== "function") throw new TypeError(`${name} must be a function. Received ${value === null ? "null" : typeof value}`);
}
function validateUndefined(value, name) {
	if (value !== void 0) throw new TypeError(`${name} must be undefined. Received ${value === null ? "null" : typeof value}`);
}
function isPromiseLike(obj) {
	return !!(obj && (typeof obj === "object" || typeof obj === "function") && typeof obj.then === "function");
}
function wrapInstanceExports(exports$1, mapFn) {
	const newExports = Object.create(null);
	Object.keys(exports$1).forEach((name) => {
		const exportValue = exports$1[name];
		Object.defineProperty(newExports, name, {
			enumerable: true,
			value: mapFn(exportValue, name)
		});
	});
	return newExports;
}
function sleepBreakIf(delay, breakIf) {
	const end = Date.now() + delay;
	let ret = false;
	while (Date.now() < end) if (breakIf()) {
		ret = true;
		break;
	}
	return ret;
}
function unsharedSlice(view, start, end) {
	return typeof SharedArrayBuffer === "function" && view.buffer instanceof SharedArrayBuffer || Object.prototype.toString.call(view.buffer.constructor) === "[object SharedArrayBuffer]" ? view.slice(start, end) : view.subarray(start, end);
}
function tryAllocate(instance, wasm64, size, mallocName) {
	if (typeof instance.exports[mallocName] !== "function" || size <= 0) return {
		wasm64,
		dataPtr: 16,
		start: wasm64 ? 32 : 24,
		end: 1024
	};
	const malloc = instance.exports[mallocName];
	const dataPtr = wasm64 ? Number(malloc(BigInt(16) + BigInt(size))) : malloc(8 + size);
	if (dataPtr === 0) throw new Error("Allocate asyncify data failed");
	return wasm64 ? {
		wasm64,
		dataPtr,
		start: dataPtr + 16,
		end: dataPtr + 16 + size
	} : {
		wasm64,
		dataPtr,
		start: dataPtr + 8,
		end: dataPtr + 8 + size
	};
}
function validateImports(imports) {
	if (imports && typeof imports !== "object") throw new TypeError("imports must be an object or undefined");
}
function fetchWasm(urlOrBuffer, imports) {
	if (typeof wx !== "undefined" && typeof __wxConfig !== "undefined") return _WebAssembly.instantiate(urlOrBuffer, imports);
	return fetch(urlOrBuffer).then((response) => response.arrayBuffer()).then((buffer) => _WebAssembly.instantiate(buffer, imports));
}
/** @public */
function load(wasmInput, imports) {
	validateImports(imports);
	imports = imports !== null && imports !== void 0 ? imports : {};
	let source;
	if (wasmInput instanceof ArrayBuffer || ArrayBuffer.isView(wasmInput)) return _WebAssembly.instantiate(wasmInput, imports);
	if (wasmInput instanceof _WebAssembly.Module) return _WebAssembly.instantiate(wasmInput, imports).then((instance) => {
		return {
			instance,
			module: wasmInput
		};
	});
	if (typeof wasmInput !== "string" && !(wasmInput instanceof URL)) throw new TypeError("Invalid source");
	if (typeof _WebAssembly.instantiateStreaming === "function") {
		let responsePromise;
		try {
			responsePromise = fetch(wasmInput);
			source = _WebAssembly.instantiateStreaming(responsePromise, imports).catch(() => {
				return fetchWasm(wasmInput, imports);
			});
		} catch (_) {
			source = fetchWasm(wasmInput, imports);
		}
	} else source = fetchWasm(wasmInput, imports);
	return source;
}
/** @public */
function asyncifyLoad(asyncify, urlOrBuffer, imports) {
	validateImports(imports);
	imports = imports !== null && imports !== void 0 ? imports : {};
	const asyncifyHelper = new Asyncify();
	imports = asyncifyHelper.wrapImports(imports);
	return load(urlOrBuffer, imports).then((source) => {
		var _a;
		const memory = source.instance.exports.memory || ((_a = imports.env) === null || _a === void 0 ? void 0 : _a.memory);
		return {
			module: source.module,
			instance: asyncifyHelper.init(memory, source.instance, asyncify)
		};
	});
}
/** @public */
function loadSync(wasmInput, imports) {
	validateImports(imports);
	imports = imports !== null && imports !== void 0 ? imports : {};
	let module$1;
	if (wasmInput instanceof ArrayBuffer || ArrayBuffer.isView(wasmInput)) module$1 = new _WebAssembly.Module(wasmInput);
	else if (wasmInput instanceof WebAssembly.Module) module$1 = wasmInput;
	else throw new TypeError("Invalid source");
	return {
		instance: new _WebAssembly.Instance(module$1, imports),
		module: module$1
	};
}
/** @public */
function asyncifyLoadSync(asyncify, buffer, imports) {
	var _a;
	validateImports(imports);
	imports = imports !== null && imports !== void 0 ? imports : {};
	const asyncifyHelper = new Asyncify();
	imports = asyncifyHelper.wrapImports(imports);
	const source = loadSync(buffer, imports);
	const memory = source.instance.exports.memory || ((_a = imports.env) === null || _a === void 0 ? void 0 : _a.memory);
	return {
		module: source.module,
		instance: asyncifyHelper.init(memory, source.instance, asyncify)
	};
}
function isPosixPathSeparator(code) {
	return code === CHAR_FORWARD_SLASH;
}
function normalizeString(path, allowAboveRoot, separator, isPathSeparator) {
	let res = "";
	let lastSegmentLength = 0;
	let lastSlash = -1;
	let dots = 0;
	let code = 0;
	for (let i = 0; i <= path.length; ++i) {
		if (i < path.length) code = path.charCodeAt(i);
		else if (isPathSeparator(code)) break;
		else code = CHAR_FORWARD_SLASH;
		if (isPathSeparator(code)) {
			if (lastSlash === i - 1 || dots === 1);
			else if (dots === 2) {
				if (res.length < 2 || lastSegmentLength !== 2 || res.charCodeAt(res.length - 1) !== CHAR_DOT || res.charCodeAt(res.length - 2) !== CHAR_DOT) {
					if (res.length > 2) {
						const lastSlashIndex = res.indexOf(separator);
						if (lastSlashIndex === -1) {
							res = "";
							lastSegmentLength = 0;
						} else {
							res = res.slice(0, lastSlashIndex);
							lastSegmentLength = res.length - 1 - res.indexOf(separator);
						}
						lastSlash = i;
						dots = 0;
						continue;
					} else if (res.length !== 0) {
						res = "";
						lastSegmentLength = 0;
						lastSlash = i;
						dots = 0;
						continue;
					}
				}
				if (allowAboveRoot) {
					res += res.length > 0 ? `${separator}..` : "..";
					lastSegmentLength = 2;
				}
			} else {
				if (res.length > 0) res += `${separator}${path.slice(lastSlash + 1, i)}`;
				else res = path.slice(lastSlash + 1, i);
				lastSegmentLength = i - lastSlash - 1;
			}
			lastSlash = i;
			dots = 0;
		} else if (code === CHAR_DOT && dots !== -1) ++dots;
		else dots = -1;
	}
	return res;
}
function resolve(...args) {
	let resolvedPath = "";
	let resolvedAbsolute = false;
	for (let i = args.length - 1; i >= -1 && !resolvedAbsolute; i--) {
		const path = i >= 0 ? args[i] : "/";
		validateString(path, "path");
		if (path.length === 0) continue;
		resolvedPath = `${path}/${resolvedPath}`;
		resolvedAbsolute = path.charCodeAt(0) === CHAR_FORWARD_SLASH;
	}
	resolvedPath = normalizeString(resolvedPath, !resolvedAbsolute, "/", isPosixPathSeparator);
	if (resolvedAbsolute) return `/${resolvedPath}`;
	return resolvedPath.length > 0 ? resolvedPath : ".";
}
function strerror(errno) {
	switch (errno) {
		case 0: return "Success";
		case 1: return "Argument list too long";
		case 2: return "Permission denied";
		case 3: return "Address in use";
		case 4: return "Address not available";
		case 5: return "Address family not supported by protocol";
		case 6: return "Resource temporarily unavailable";
		case 7: return "Operation already in progress";
		case 8: return "Bad file descriptor";
		case 9: return "Bad message";
		case 10: return "Resource busy";
		case 11: return "Operation canceled";
		case 12: return "No child process";
		case 13: return "Connection aborted";
		case 14: return "Connection refused";
		case 15: return "Connection reset by peer";
		case 16: return "Resource deadlock would occur";
		case 17: return "Destination address required";
		case 18: return "Domain error";
		case 19: return "Quota exceeded";
		case 20: return "File exists";
		case 21: return "Bad address";
		case 22: return "File too large";
		case 23: return "Host is unreachable";
		case 24: return "Identifier removed";
		case 25: return "Illegal byte sequence";
		case 26: return "Operation in progress";
		case 27: return "Interrupted system call";
		case 28: return "Invalid argument";
		case 29: return "I/O error";
		case 30: return "Socket is connected";
		case 31: return "Is a directory";
		case 32: return "Symbolic link loop";
		case 33: return "No file descriptors available";
		case 34: return "Too many links";
		case 35: return "Message too large";
		case 36: return "Multihop attempted";
		case 37: return "Filename too long";
		case 38: return "Network is down";
		case 39: return "Connection reset by network";
		case 40: return "Network unreachable";
		case 41: return "Too many files open in system";
		case 42: return "No buffer space available";
		case 43: return "No such device";
		case 44: return "No such file or directory";
		case 45: return "Exec format error";
		case 46: return "No locks available";
		case 47: return "Link has been severed";
		case 48: return "Out of memory";
		case 49: return "No message of the desired type";
		case 50: return "Protocol not available";
		case 51: return "No space left on device";
		case 52: return "Function not implemented";
		case 53: return "Socket not connected";
		case 54: return "Not a directory";
		case 55: return "Directory not empty";
		case 56: return "State not recoverable";
		case 57: return "Not a socket";
		case 58: return "Not supported";
		case 59: return "Not a tty";
		case 60: return "No such device or address";
		case 61: return "Value too large for data type";
		case 62: return "Previous owner died";
		case 63: return "Operation not permitted";
		case 64: return "Broken pipe";
		case 65: return "Protocol error";
		case 66: return "Protocol not supported";
		case 67: return "Protocol wrong type for socket";
		case 68: return "Result not representable";
		case 69: return "Read-only file system";
		case 70: return "Invalid seek";
		case 71: return "No such process";
		case 72: return "Stale file handle";
		case 73: return "Operation timed out";
		case 74: return "Text file busy";
		case 75: return "Cross-device link";
		case 76: return "Capabilities insufficient";
		default: return "Unknown error";
	}
}
function getRights(stdio, fd, flags, type) {
	const ret = {
		base: BigInt(0),
		inheriting: BigInt(0)
	};
	if (type === 0) throw new WasiError("Unknown file type", 28);
	switch (type) {
		case 4:
			ret.base = REGULAR_FILE_BASE;
			ret.inheriting = REGULAR_FILE_INHERITING;
			break;
		case 3:
			ret.base = DIRECTORY_BASE;
			ret.inheriting = DIRECTORY_INHERITING;
			break;
		case 6:
		case 5:
			ret.base = SOCKET_BASE;
			ret.inheriting = SOCKET_INHERITING;
			break;
		case 2:
			if (stdio.indexOf(fd) !== -1) {
				ret.base = TTY_BASE;
				ret.inheriting = TTY_INHERITING;
			} else {
				ret.base = CHARACTER_DEVICE_BASE;
				ret.inheriting = CHARACTER_DEVICE_INHERITING;
			}
			break;
		case 1:
			ret.base = BLOCK_DEVICE_BASE;
			ret.inheriting = BLOCK_DEVICE_INHERITING;
			break;
		default:
			ret.base = BigInt(0);
			ret.inheriting = BigInt(0);
	}
	const read_or_write_only = flags & 3;
	if (read_or_write_only === 0) ret.base &= ~WasiRights.FD_WRITE;
	else if (read_or_write_only === 1) ret.base &= ~WasiRights.FD_READ;
	return ret;
}
function concatBuffer(buffers, size) {
	let total = 0;
	if (typeof size === "number" && size >= 0) total = size;
	else for (let i = 0; i < buffers.length; i++) {
		const buffer = buffers[i];
		total += buffer.length;
	}
	let pos = 0;
	const ret = new Uint8Array(total);
	for (let i = 0; i < buffers.length; i++) {
		const buffer = buffers[i];
		ret.set(buffer, pos);
		pos += buffer.length;
	}
	return ret;
}
function toFileType(stat) {
	if (stat.isBlockDevice()) return 1;
	if (stat.isCharacterDevice()) return 2;
	if (stat.isDirectory()) return 3;
	if (stat.isSocket()) return 6;
	if (stat.isFile()) return 4;
	if (stat.isSymbolicLink()) return 7;
	return 0;
}
function toFileStat(view, buf, stat) {
	view.setBigUint64(buf, stat.dev, true);
	view.setBigUint64(buf + 8, stat.ino, true);
	view.setBigUint64(buf + 16, BigInt(toFileType(stat)), true);
	view.setBigUint64(buf + 24, stat.nlink, true);
	view.setBigUint64(buf + 32, stat.size, true);
	view.setBigUint64(buf + 40, stat.atimeMs * BigInt(1e6), true);
	view.setBigUint64(buf + 48, stat.mtimeMs * BigInt(1e6), true);
	view.setBigUint64(buf + 56, stat.ctimeMs * BigInt(1e6), true);
}
/** @public */
function extendMemory(memory) {
	if (Object.getPrototypeOf(memory) === _WebAssembly.Memory.prototype) Object.setPrototypeOf(memory, Memory.prototype);
	return memory;
}
function checkWebAssemblyFunction() {
	const WebAssemblyFunction = _WebAssembly.Function;
	if (typeof WebAssemblyFunction !== "function") throw new Error("WebAssembly.Function is not supported in this environment. If you are using V8 based browser like Chrome, try to specify --js-flags=\"--wasm-staging --experimental-wasm-stack-switching\"");
	return WebAssemblyFunction;
}
/** @public */
function wrapAsyncImport(f, parameterType, returnType) {
	const WebAssemblyFunction = checkWebAssemblyFunction();
	if (typeof f !== "function") throw new TypeError("Function required");
	const parameters = parameterType.slice(0);
	parameters.unshift("externref");
	return new WebAssemblyFunction({
		parameters,
		results: returnType
	}, f, { suspending: "first" });
}
/** @public */
function wrapAsyncExport(f) {
	const WebAssemblyFunction = checkWebAssemblyFunction();
	if (typeof f !== "function") throw new TypeError("Function required");
	return new WebAssemblyFunction({
		parameters: [...WebAssemblyFunction.type(f).parameters.slice(1)],
		results: ["externref"]
	}, f, { promising: "first" });
}
/** @public */
function wrapExports(exports$1, needWrap) {
	return wrapInstanceExports(exports$1, (exportValue, name) => {
		let ignore = typeof exportValue !== "function";
		if (Array.isArray(needWrap)) ignore = ignore || needWrap.indexOf(name) === -1;
		return ignore ? exportValue : wrapAsyncExport(exportValue);
	});
}
function copyMemory(targets, src) {
	if (targets.length === 0 || src.length === 0) return 0;
	let copied = 0;
	let left = src.length - copied;
	for (let i = 0; i < targets.length; ++i) {
		const target = targets[i];
		if (left < target.length) {
			target.set(src.subarray(copied, copied + left), 0);
			copied += left;
			left = 0;
			return copied;
		}
		target.set(src.subarray(copied, copied + target.length), 0);
		copied += target.length;
		left -= target.length;
	}
	return copied;
}
function getMemory(wasi) {
	return _memory.get(wasi);
}
function getFs(wasi) {
	const fs$1 = _fs.get(wasi);
	if (!fs$1) throw new Error("filesystem is unavailable");
	return fs$1;
}
function handleError(err) {
	if (err instanceof WasiError) {
		if (process.env.NODE_ENV !== "production") console.warn(err);
		return err.errno;
	}
	switch (err.code) {
		case "ENOENT": return 44;
		case "EBADF": return 8;
		case "EINVAL": return 28;
		case "EPERM": return 63;
		case "EPROTO": return 65;
		case "EEXIST": return 20;
		case "ENOTDIR": return 54;
		case "EMFILE": return 33;
		case "EACCES": return 2;
		case "EISDIR": return 31;
		case "ENOTEMPTY": return 55;
		case "ENOSYS": return 52;
	}
	throw err;
}
function defineName(name, f) {
	Object.defineProperty(f, "name", { value: name });
	return f;
}
function tryCall(f, wasi, args) {
	let r;
	try {
		r = f.apply(wasi, args);
	} catch (err) {
		return handleError(err);
	}
	if (isPromiseLike(r)) return r.then((_) => _, handleError);
	return r;
}
function syscallWrap(self$1, name, f) {
	let debug = false;
	const NODE_DEBUG_NATIVE = (() => {
		try {
			return process.env.NODE_DEBUG_NATIVE;
		} catch (_) {
			return;
		}
	})();
	if (typeof NODE_DEBUG_NATIVE === "string" && NODE_DEBUG_NATIVE.split(",").includes("wasi")) debug = true;
	return debug ? defineName(name, function() {
		const args = Array.prototype.slice.call(arguments);
		let debugArgs = [`${name}(${Array.from({ length: arguments.length }).map(() => "%d").join(", ")})`];
		debugArgs = debugArgs.concat(args);
		console.debug.apply(console, debugArgs);
		return tryCall(f, self$1, args);
	}) : defineName(name, function() {
		return tryCall(f, self$1, arguments);
	});
}
function resolvePathSync(fs$1, fileDescriptor, path, flags) {
	let resolvedPath = resolve(fileDescriptor.realPath, path);
	if ((flags & 1) === 1) try {
		resolvedPath = fs$1.readlinkSync(resolvedPath);
	} catch (err) {
		if (err.code !== "EINVAL" && err.code !== "ENOENT") throw err;
	}
	return resolvedPath;
}
async function resolvePathAsync(fs$1, fileDescriptor, path, flags) {
	let resolvedPath = resolve(fileDescriptor.realPath, path);
	if ((flags & 1) === 1) try {
		resolvedPath = await fs$1.promises.readlink(resolvedPath);
	} catch (err) {
		if (err.code !== "EINVAL" && err.code !== "ENOENT") throw err;
	}
	return resolvedPath;
}
function readStdin() {
	const value = window.prompt();
	if (value === null) return new Uint8Array();
	return new TextEncoder().encode(value + "\n");
}
function validateFstFlagsOrReturn(flags) {
	return Boolean(flags & -16) || (flags & 3) === 3 || (flags & 12) === 12;
}
function validateOptions(options) {
	var _a;
	validateObject(options, "options");
	let _WASI;
	if (options.version !== void 0) {
		validateString(options.version, "options.version");
		switch (options.version) {
			case "unstable":
				_WASI = WASI$1;
				this[kBindingName] = "wasi_unstable";
				break;
			case "preview1":
				_WASI = WASI$1;
				this[kBindingName] = "wasi_snapshot_preview1";
				break;
			default: throw new TypeError(`unsupported WASI version "${options.version}"`);
		}
	} else {
		_WASI = WASI$1;
		this[kBindingName] = "wasi_snapshot_preview1";
	}
	if (options.args !== void 0) validateArray(options.args, "options.args");
	const args = ((_a = options.args) !== null && _a !== void 0 ? _a : []).map(String);
	const env = [];
	if (options.env !== void 0) {
		validateObject(options.env, "options.env");
		Object.entries(options.env).forEach(({ 0: key, 1: value }) => {
			if (value !== void 0) env.push(`${key}=${value}`);
		});
	}
	const preopens = [];
	if (options.preopens !== void 0) {
		validateObject(options.preopens, "options.preopens");
		Object.entries(options.preopens).forEach(({ 0: key, 1: value }) => preopens.push({
			mappedPath: String(key),
			realPath: String(value)
		}));
	}
	if (preopens.length > 0) {
		if (options.fs === void 0) throw new Error("filesystem is disabled, can not preopen directory");
		try {
			validateObject(options.fs, "options.fs");
		} catch (_) {
			throw new TypeError("Node.js fs like implementation is not provided");
		}
	}
	if (options.print !== void 0) validateFunction(options.print, "options.print");
	if (options.printErr !== void 0) validateFunction(options.printErr, "options.printErr");
	if (options.returnOnExit !== void 0) validateBoolean(options.returnOnExit, "options.returnOnExit");
	return {
		args,
		env,
		preopens,
		stdio: [
			0,
			1,
			2
		],
		_WASI
	};
}
function initWASI(setMemory, wrap$2) {
	this[kSetMemory] = setMemory;
	this.wasiImport = wrap$2;
	this[kStarted] = false;
	this[kExitCode] = 0;
	this[kInstance] = void 0;
}
function wasiReturnOnProcExit(rval) {
	this[kExitCode] = rval;
	throw kExitCode;
}
/** @public */
async function createAsyncWASI(options = kEmptyObject) {
	const _this = Object.create(WASI$2.prototype);
	const { args, env, preopens, stdio, _WASI } = validateOptions.call(_this, options);
	if (options.asyncify !== void 0) {
		validateObject(options.asyncify, "options.asyncify");
		validateFunction(options.asyncify.wrapImportFunction, "options.asyncify.wrapImportFunction");
	}
	const wrap$2 = await _WASI.createAsync(args, env, preopens, stdio, options.fs, options.print, options.printErr, options.asyncify);
	const setMemory = wrap$2._setMemory;
	delete wrap$2._setMemory;
	initWASI.call(_this, setMemory, wrap$2);
	if (options.returnOnExit) wrap$2.proc_exit = wasiReturnOnProcExit.bind(_this);
	return _this;
}
var _WebAssembly, ignoreNames, Asyncify, CHAR_DOT, CHAR_FORWARD_SLASH, FD_DATASYNC, FD_READ, FD_SEEK, FD_FDSTAT_SET_FLAGS, FD_SYNC, FD_TELL, FD_WRITE, FD_ADVISE, FD_ALLOCATE, PATH_CREATE_DIRECTORY, PATH_CREATE_FILE, PATH_LINK_SOURCE, PATH_LINK_TARGET, PATH_OPEN, FD_READDIR, PATH_READLINK, PATH_RENAME_SOURCE, PATH_RENAME_TARGET, PATH_FILESTAT_GET, PATH_FILESTAT_SET_SIZE, PATH_FILESTAT_SET_TIMES, FD_FILESTAT_GET, FD_FILESTAT_SET_SIZE, FD_FILESTAT_SET_TIMES, PATH_SYMLINK, PATH_REMOVE_DIRECTORY, PATH_UNLINK_FILE, POLL_FD_READWRITE, SOCK_SHUTDOWN, SOCK_ACCEPT, WasiRights, WasiError, RIGHTS_ALL, BLOCK_DEVICE_BASE, BLOCK_DEVICE_INHERITING, CHARACTER_DEVICE_BASE, CHARACTER_DEVICE_INHERITING, REGULAR_FILE_BASE, REGULAR_FILE_INHERITING, DIRECTORY_BASE, DIRECTORY_INHERITING, SOCKET_BASE, SOCKET_INHERITING, TTY_BASE, TTY_INHERITING, FileDescriptor, StandardOutput, FileDescriptorTable, SyncTable, AsyncTable, WebAssemblyMemory, Memory, _memory, _wasi, _fs, encoder, decoder, INT64_MAX, WASI$1, kEmptyObject, kExitCode, kSetMemory, kStarted, kInstance, kBindingName, WASI$2;
var init_wasm_util_esm_bundler = __esm({ "../../node_modules/.pnpm/@tybys+wasm-util@0.10.1/node_modules/@tybys/wasm-util/dist/wasm-util.esm-bundler.js": (() => {
	_WebAssembly = typeof WebAssembly !== "undefined" ? WebAssembly : typeof WXWebAssembly !== "undefined" ? WXWebAssembly : void 0;
	if (!_WebAssembly) throw new Error("WebAssembly is not supported in this environment");
	ignoreNames = [
		"asyncify_get_state",
		"asyncify_start_rewind",
		"asyncify_start_unwind",
		"asyncify_stop_rewind",
		"asyncify_stop_unwind"
	];
	Asyncify = class {
		constructor() {
			this.value = void 0;
			this.exports = void 0;
			this.dataPtr = 0;
		}
		init(memory, instance, options) {
			var _a, _b;
			if (this.exports) throw new Error("Asyncify has been initialized");
			if (!(memory instanceof _WebAssembly.Memory)) throw new TypeError("Require WebAssembly.Memory object");
			const exports$1 = instance.exports;
			for (let i = 0; i < ignoreNames.length; ++i) if (typeof exports$1[ignoreNames[i]] !== "function") throw new TypeError("Invalid asyncify wasm");
			let address;
			const wasm64 = Boolean(options.wasm64);
			if (!options.tryAllocate) address = {
				wasm64,
				dataPtr: 16,
				start: wasm64 ? 32 : 24,
				end: 1024
			};
			else if (options.tryAllocate === true) address = tryAllocate(instance, wasm64, 4096, "malloc");
			else address = tryAllocate(instance, wasm64, (_a = options.tryAllocate.size) !== null && _a !== void 0 ? _a : 4096, (_b = options.tryAllocate.name) !== null && _b !== void 0 ? _b : "malloc");
			this.dataPtr = address.dataPtr;
			if (wasm64) new BigInt64Array(memory.buffer, this.dataPtr).set([BigInt(address.start), BigInt(address.end)]);
			else new Int32Array(memory.buffer, this.dataPtr).set([address.start, address.end]);
			this.exports = this.wrapExports(exports$1, options.wrapExports);
			const asyncifiedInstance = Object.create(_WebAssembly.Instance.prototype);
			Object.defineProperty(asyncifiedInstance, "exports", { value: this.exports });
			return asyncifiedInstance;
		}
		assertState() {
			if (this.exports.asyncify_get_state() !== 0) throw new Error("Asyncify state error");
		}
		wrapImportFunction(f) {
			const _this = this;
			return (function() {
				while (_this.exports.asyncify_get_state() === 2) {
					_this.exports.asyncify_stop_rewind();
					return _this.value;
				}
				_this.assertState();
				const v = f.apply(this, arguments);
				if (!isPromiseLike(v)) return v;
				_this.exports.asyncify_start_unwind(_this.dataPtr);
				_this.value = v;
			});
		}
		wrapImports(imports) {
			const importObject = {};
			Object.keys(imports).forEach((k) => {
				const mod = imports[k];
				const newModule = {};
				Object.keys(mod).forEach((name) => {
					const importValue = mod[name];
					if (typeof importValue === "function") newModule[name] = this.wrapImportFunction(importValue);
					else newModule[name] = importValue;
				});
				importObject[k] = newModule;
			});
			return importObject;
		}
		wrapExportFunction(f) {
			const _this = this;
			return (async function() {
				_this.assertState();
				let ret = f.apply(this, arguments);
				while (_this.exports.asyncify_get_state() === 1) {
					_this.exports.asyncify_stop_unwind();
					_this.value = await _this.value;
					_this.assertState();
					_this.exports.asyncify_start_rewind(_this.dataPtr);
					ret = f.call(this);
				}
				_this.assertState();
				return ret;
			});
		}
		wrapExports(exports$1, needWrap) {
			return wrapInstanceExports(exports$1, (exportValue, name) => {
				let ignore = ignoreNames.indexOf(name) !== -1 || typeof exportValue !== "function";
				if (Array.isArray(needWrap)) ignore = ignore || needWrap.indexOf(name) === -1;
				return ignore ? exportValue : this.wrapExportFunction(exportValue);
			});
		}
	};
	CHAR_DOT = 46;
	CHAR_FORWARD_SLASH = 47;
	FD_DATASYNC = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(0);
	FD_READ = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(1);
	FD_SEEK = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(2);
	FD_FDSTAT_SET_FLAGS = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(3);
	FD_SYNC = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(4);
	FD_TELL = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(5);
	FD_WRITE = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(6);
	FD_ADVISE = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(7);
	FD_ALLOCATE = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(8);
	PATH_CREATE_DIRECTORY = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(9);
	PATH_CREATE_FILE = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(10);
	PATH_LINK_SOURCE = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(11);
	PATH_LINK_TARGET = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(12);
	PATH_OPEN = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(13);
	FD_READDIR = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(14);
	PATH_READLINK = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(15);
	PATH_RENAME_SOURCE = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(16);
	PATH_RENAME_TARGET = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(17);
	PATH_FILESTAT_GET = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(18);
	PATH_FILESTAT_SET_SIZE = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(19);
	PATH_FILESTAT_SET_TIMES = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(20);
	FD_FILESTAT_GET = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(21);
	FD_FILESTAT_SET_SIZE = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(22);
	FD_FILESTAT_SET_TIMES = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(23);
	PATH_SYMLINK = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(24);
	PATH_REMOVE_DIRECTORY = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(25);
	PATH_UNLINK_FILE = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(26);
	POLL_FD_READWRITE = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(27);
	SOCK_SHUTDOWN = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(28);
	SOCK_ACCEPT = /* @__PURE__ */ BigInt(1) << /* @__PURE__ */ BigInt(29);
	WasiRights = {
		FD_DATASYNC,
		FD_READ,
		FD_SEEK,
		FD_FDSTAT_SET_FLAGS,
		FD_SYNC,
		FD_TELL,
		FD_WRITE,
		FD_ADVISE,
		FD_ALLOCATE,
		PATH_CREATE_DIRECTORY,
		PATH_CREATE_FILE,
		PATH_LINK_SOURCE,
		PATH_LINK_TARGET,
		PATH_OPEN,
		FD_READDIR,
		PATH_READLINK,
		PATH_RENAME_SOURCE,
		PATH_RENAME_TARGET,
		PATH_FILESTAT_GET,
		PATH_FILESTAT_SET_SIZE,
		PATH_FILESTAT_SET_TIMES,
		FD_FILESTAT_GET,
		FD_FILESTAT_SET_SIZE,
		FD_FILESTAT_SET_TIMES,
		PATH_SYMLINK,
		PATH_REMOVE_DIRECTORY,
		PATH_UNLINK_FILE,
		POLL_FD_READWRITE,
		SOCK_SHUTDOWN,
		SOCK_ACCEPT
	};
	WasiError = class extends Error {
		constructor(message, errno) {
			super(message);
			this.errno = errno;
		}
		getErrorMessage() {
			return strerror(this.errno);
		}
	};
	Object.defineProperty(WasiError.prototype, "name", {
		configurable: true,
		writable: true,
		value: "WasiError"
	});
	RIGHTS_ALL = WasiRights.FD_DATASYNC | WasiRights.FD_READ | WasiRights.FD_SEEK | WasiRights.FD_FDSTAT_SET_FLAGS | WasiRights.FD_SYNC | WasiRights.FD_TELL | WasiRights.FD_WRITE | WasiRights.FD_ADVISE | WasiRights.FD_ALLOCATE | WasiRights.PATH_CREATE_DIRECTORY | WasiRights.PATH_CREATE_FILE | WasiRights.PATH_LINK_SOURCE | WasiRights.PATH_LINK_TARGET | WasiRights.PATH_OPEN | WasiRights.FD_READDIR | WasiRights.PATH_READLINK | WasiRights.PATH_RENAME_SOURCE | WasiRights.PATH_RENAME_TARGET | WasiRights.PATH_FILESTAT_GET | WasiRights.PATH_FILESTAT_SET_SIZE | WasiRights.PATH_FILESTAT_SET_TIMES | WasiRights.FD_FILESTAT_GET | WasiRights.FD_FILESTAT_SET_TIMES | WasiRights.FD_FILESTAT_SET_SIZE | WasiRights.PATH_SYMLINK | WasiRights.PATH_UNLINK_FILE | WasiRights.PATH_REMOVE_DIRECTORY | WasiRights.POLL_FD_READWRITE | WasiRights.SOCK_SHUTDOWN | WasiRights.SOCK_ACCEPT;
	BLOCK_DEVICE_BASE = RIGHTS_ALL;
	BLOCK_DEVICE_INHERITING = RIGHTS_ALL;
	CHARACTER_DEVICE_BASE = RIGHTS_ALL;
	CHARACTER_DEVICE_INHERITING = RIGHTS_ALL;
	REGULAR_FILE_BASE = WasiRights.FD_DATASYNC | WasiRights.FD_READ | WasiRights.FD_SEEK | WasiRights.FD_FDSTAT_SET_FLAGS | WasiRights.FD_SYNC | WasiRights.FD_TELL | WasiRights.FD_WRITE | WasiRights.FD_ADVISE | WasiRights.FD_ALLOCATE | WasiRights.FD_FILESTAT_GET | WasiRights.FD_FILESTAT_SET_SIZE | WasiRights.FD_FILESTAT_SET_TIMES | WasiRights.POLL_FD_READWRITE;
	REGULAR_FILE_INHERITING = /* @__PURE__ */ BigInt(0);
	DIRECTORY_BASE = WasiRights.FD_FDSTAT_SET_FLAGS | WasiRights.FD_SYNC | WasiRights.FD_ADVISE | WasiRights.PATH_CREATE_DIRECTORY | WasiRights.PATH_CREATE_FILE | WasiRights.PATH_LINK_SOURCE | WasiRights.PATH_LINK_TARGET | WasiRights.PATH_OPEN | WasiRights.FD_READDIR | WasiRights.PATH_READLINK | WasiRights.PATH_RENAME_SOURCE | WasiRights.PATH_RENAME_TARGET | WasiRights.PATH_FILESTAT_GET | WasiRights.PATH_FILESTAT_SET_SIZE | WasiRights.PATH_FILESTAT_SET_TIMES | WasiRights.FD_FILESTAT_GET | WasiRights.FD_FILESTAT_SET_TIMES | WasiRights.PATH_SYMLINK | WasiRights.PATH_UNLINK_FILE | WasiRights.PATH_REMOVE_DIRECTORY | WasiRights.POLL_FD_READWRITE;
	DIRECTORY_INHERITING = DIRECTORY_BASE | REGULAR_FILE_BASE;
	SOCKET_BASE = WasiRights.FD_READ | WasiRights.FD_FDSTAT_SET_FLAGS | WasiRights.FD_WRITE | WasiRights.FD_FILESTAT_GET | WasiRights.POLL_FD_READWRITE | WasiRights.SOCK_SHUTDOWN;
	SOCKET_INHERITING = RIGHTS_ALL;
	TTY_BASE = WasiRights.FD_READ | WasiRights.FD_FDSTAT_SET_FLAGS | WasiRights.FD_WRITE | WasiRights.FD_FILESTAT_GET | WasiRights.POLL_FD_READWRITE;
	TTY_INHERITING = /* @__PURE__ */ BigInt(0);
	FileDescriptor = class {
		constructor(id, fd, path, realPath, type, rightsBase, rightsInheriting, preopen) {
			this.id = id;
			this.fd = fd;
			this.path = path;
			this.realPath = realPath;
			this.type = type;
			this.rightsBase = rightsBase;
			this.rightsInheriting = rightsInheriting;
			this.preopen = preopen;
			this.pos = BigInt(0);
			this.size = BigInt(0);
		}
		seek(offset, whence) {
			if (whence === 0) this.pos = BigInt(offset);
			else if (whence === 1) this.pos += BigInt(offset);
			else if (whence === 2) this.pos = BigInt(this.size) - BigInt(offset);
			else throw new WasiError("Unknown whence", 29);
			return this.pos;
		}
	};
	StandardOutput = class extends FileDescriptor {
		constructor(log, id, fd, path, realPath, type, rightsBase, rightsInheriting, preopen) {
			super(id, fd, path, realPath, type, rightsBase, rightsInheriting, preopen);
			this._log = log;
			this._buf = null;
		}
		write(buffer) {
			const originalBuffer = buffer;
			if (this._buf) {
				buffer = concatBuffer([this._buf, buffer]);
				this._buf = null;
			}
			if (buffer.indexOf(10) === -1) {
				this._buf = buffer;
				return originalBuffer.byteLength;
			}
			let written = 0;
			let lastBegin = 0;
			let index;
			while ((index = buffer.indexOf(10, written)) !== -1) {
				const str = new TextDecoder().decode(buffer.subarray(lastBegin, index));
				this._log(str);
				written += index - lastBegin + 1;
				lastBegin = index + 1;
			}
			if (written < buffer.length) this._buf = buffer.slice(written);
			return originalBuffer.byteLength;
		}
	};
	FileDescriptorTable = class {
		constructor(options) {
			this.used = 0;
			this.size = options.size;
			this.fds = Array(options.size);
			this.stdio = [
				options.in,
				options.out,
				options.err
			];
			this.print = options.print;
			this.printErr = options.printErr;
			this.insertStdio(options.in, 0, "<stdin>");
			this.insertStdio(options.out, 1, "<stdout>");
			this.insertStdio(options.err, 2, "<stderr>");
		}
		insertStdio(fd, expected, name) {
			const type = 2;
			const { base, inheriting } = getRights(this.stdio, fd, 2, type);
			const wrap$2 = this.insert(fd, name, name, type, base, inheriting, 0);
			if (wrap$2.id !== expected) throw new WasiError(`id: ${wrap$2.id} !== expected: ${expected}`, 8);
			return wrap$2;
		}
		insert(fd, mappedPath, realPath, type, rightsBase, rightsInheriting, preopen) {
			var _a, _b;
			let index = -1;
			if (this.used >= this.size) {
				const newSize = this.size * 2;
				this.fds.length = newSize;
				index = this.size;
				this.size = newSize;
			} else for (let i = 0; i < this.size; ++i) if (this.fds[i] == null) {
				index = i;
				break;
			}
			let entry;
			if (mappedPath === "<stdout>") entry = new StandardOutput((_a = this.print) !== null && _a !== void 0 ? _a : console.log, index, fd, mappedPath, realPath, type, rightsBase, rightsInheriting, preopen);
			else if (mappedPath === "<stderr>") entry = new StandardOutput((_b = this.printErr) !== null && _b !== void 0 ? _b : console.error, index, fd, mappedPath, realPath, type, rightsBase, rightsInheriting, preopen);
			else entry = new FileDescriptor(index, fd, mappedPath, realPath, type, rightsBase, rightsInheriting, preopen);
			this.fds[index] = entry;
			this.used++;
			return entry;
		}
		get(id, base, inheriting) {
			if (id >= this.size) throw new WasiError("Invalid fd", 8);
			const entry = this.fds[id];
			if (!entry || entry.id !== id) throw new WasiError("Bad file descriptor", 8);
			if ((~entry.rightsBase & base) !== BigInt(0) || (~entry.rightsInheriting & inheriting) !== BigInt(0)) throw new WasiError("Capabilities insufficient", 76);
			return entry;
		}
		remove(id) {
			if (id >= this.size) throw new WasiError("Invalid fd", 8);
			const entry = this.fds[id];
			if (!entry || entry.id !== id) throw new WasiError("Bad file descriptor", 8);
			this.fds[id] = void 0;
			this.used--;
		}
	};
	SyncTable = class extends FileDescriptorTable {
		constructor(options) {
			super(options);
			this.fs = options.fs;
		}
		getFileTypeByFd(fd) {
			const stats = this.fs.fstatSync(fd, { bigint: true });
			return toFileType(stats);
		}
		insertPreopen(fd, mappedPath, realPath) {
			const type = this.getFileTypeByFd(fd);
			if (type !== 3) throw new WasiError(`Preopen not dir: ["${mappedPath}", "${realPath}"]`, 54);
			const result = getRights(this.stdio, fd, 0, type);
			return this.insert(fd, mappedPath, realPath, type, result.base, result.inheriting, 1);
		}
		renumber(dst, src) {
			if (dst === src) return;
			if (dst >= this.size || src >= this.size) throw new WasiError("Invalid fd", 8);
			const dstEntry = this.fds[dst];
			const srcEntry = this.fds[src];
			if (!dstEntry || !srcEntry || dstEntry.id !== dst || srcEntry.id !== src) throw new WasiError("Invalid fd", 8);
			this.fs.closeSync(dstEntry.fd);
			this.fds[dst] = this.fds[src];
			this.fds[dst].id = dst;
			this.fds[src] = void 0;
			this.used--;
		}
	};
	AsyncTable = class extends FileDescriptorTable {
		constructor(options) {
			super(options);
		}
		async getFileTypeByFd(fd) {
			const stats = await fd.stat({ bigint: true });
			return toFileType(stats);
		}
		async insertPreopen(fd, mappedPath, realPath) {
			const type = await this.getFileTypeByFd(fd);
			if (type !== 3) throw new WasiError(`Preopen not dir: ["${mappedPath}", "${realPath}"]`, 54);
			const result = getRights(this.stdio, fd.fd, 0, type);
			return this.insert(fd, mappedPath, realPath, type, result.base, result.inheriting, 1);
		}
		async renumber(dst, src) {
			if (dst === src) return;
			if (dst >= this.size || src >= this.size) throw new WasiError("Invalid fd", 8);
			const dstEntry = this.fds[dst];
			const srcEntry = this.fds[src];
			if (!dstEntry || !srcEntry || dstEntry.id !== dst || srcEntry.id !== src) throw new WasiError("Invalid fd", 8);
			await dstEntry.fd.close();
			this.fds[dst] = this.fds[src];
			this.fds[dst].id = dst;
			this.fds[src] = void 0;
			this.used--;
		}
	};
	WebAssemblyMemory = /* @__PURE__ */ (function() {
		return _WebAssembly.Memory;
	})();
	Memory = class extends WebAssemblyMemory {
		constructor(descriptor) {
			super(descriptor);
		}
		get HEAP8() {
			return new Int8Array(super.buffer);
		}
		get HEAPU8() {
			return new Uint8Array(super.buffer);
		}
		get HEAP16() {
			return new Int16Array(super.buffer);
		}
		get HEAPU16() {
			return new Uint16Array(super.buffer);
		}
		get HEAP32() {
			return new Int32Array(super.buffer);
		}
		get HEAPU32() {
			return new Uint32Array(super.buffer);
		}
		get HEAP64() {
			return new BigInt64Array(super.buffer);
		}
		get HEAPU64() {
			return new BigUint64Array(super.buffer);
		}
		get HEAPF32() {
			return new Float32Array(super.buffer);
		}
		get HEAPF64() {
			return new Float64Array(super.buffer);
		}
		get view() {
			return new DataView(super.buffer);
		}
	};
	_memory = /* @__PURE__ */ new WeakMap();
	_wasi = /* @__PURE__ */ new WeakMap();
	_fs = /* @__PURE__ */ new WeakMap();
	encoder = /* @__PURE__ */ new TextEncoder();
	decoder = /* @__PURE__ */ new TextDecoder();
	INT64_MAX = (BigInt(1) << BigInt(63)) - BigInt(1);
	WASI$1 = class WASI$1 {
		constructor(args, env, fds, asyncFs, fs$1, asyncify) {
			this.args_get = syscallWrap(this, "args_get", function(argv, argv_buf) {
				argv = Number(argv);
				argv_buf = Number(argv_buf);
				if (argv === 0 || argv_buf === 0) return 28;
				const { HEAPU8, view } = getMemory(this);
				const args$1 = _wasi.get(this).args;
				for (let i = 0; i < args$1.length; ++i) {
					const arg = args$1[i];
					view.setInt32(argv, argv_buf, true);
					argv += 4;
					const data = encoder.encode(arg + "\0");
					HEAPU8.set(data, argv_buf);
					argv_buf += data.length;
				}
				return 0;
			});
			this.args_sizes_get = syscallWrap(this, "args_sizes_get", function(argc, argv_buf_size) {
				argc = Number(argc);
				argv_buf_size = Number(argv_buf_size);
				if (argc === 0 || argv_buf_size === 0) return 28;
				const { view } = getMemory(this);
				const args$1 = _wasi.get(this).args;
				view.setUint32(argc, args$1.length, true);
				view.setUint32(argv_buf_size, encoder.encode(args$1.join("\0") + "\0").length, true);
				return 0;
			});
			this.environ_get = syscallWrap(this, "environ_get", function(environ, environ_buf) {
				environ = Number(environ);
				environ_buf = Number(environ_buf);
				if (environ === 0 || environ_buf === 0) return 28;
				const { HEAPU8, view } = getMemory(this);
				const env$1 = _wasi.get(this).env;
				for (let i = 0; i < env$1.length; ++i) {
					const pair = env$1[i];
					view.setInt32(environ, environ_buf, true);
					environ += 4;
					const data = encoder.encode(pair + "\0");
					HEAPU8.set(data, environ_buf);
					environ_buf += data.length;
				}
				return 0;
			});
			this.environ_sizes_get = syscallWrap(this, "environ_sizes_get", function(len, buflen) {
				len = Number(len);
				buflen = Number(buflen);
				if (len === 0 || buflen === 0) return 28;
				const { view } = getMemory(this);
				const wasi = _wasi.get(this);
				view.setUint32(len, wasi.env.length, true);
				view.setUint32(buflen, encoder.encode(wasi.env.join("\0") + "\0").length, true);
				return 0;
			});
			this.clock_res_get = syscallWrap(this, "clock_res_get", function(id, resolution) {
				resolution = Number(resolution);
				if (resolution === 0) return 28;
				const { view } = getMemory(this);
				switch (id) {
					case 0:
						view.setBigUint64(resolution, BigInt(1e6), true);
						return 0;
					case 1:
					case 2:
					case 3:
						view.setBigUint64(resolution, BigInt(1e3), true);
						return 0;
					default: return 28;
				}
			});
			this.clock_time_get = syscallWrap(this, "clock_time_get", function(id, _percision, time) {
				time = Number(time);
				if (time === 0) return 28;
				const { view } = getMemory(this);
				switch (id) {
					case 0:
						view.setBigUint64(time, BigInt(Date.now()) * BigInt(1e6), true);
						return 0;
					case 1:
					case 2:
					case 3: {
						const t = performance.now() / 1e3;
						const s = Math.trunc(t);
						const ms = Math.floor((t - s) * 1e3);
						const result = BigInt(s) * BigInt(1e9) + BigInt(ms) * BigInt(1e6);
						view.setBigUint64(time, result, true);
						return 0;
					}
					default: return 28;
				}
			});
			this.fd_advise = syscallWrap(this, "fd_advise", function(_fd, _offset, _len, _advice) {
				return 52;
			});
			this.fd_fdstat_get = syscallWrap(this, "fd_fdstat_get", function(fd, fdstat) {
				fdstat = Number(fdstat);
				if (fdstat === 0) return 28;
				const fileDescriptor = _wasi.get(this).fds.get(fd, BigInt(0), BigInt(0));
				const { view } = getMemory(this);
				view.setUint16(fdstat, fileDescriptor.type, true);
				view.setUint16(fdstat + 2, 0, true);
				view.setBigUint64(fdstat + 8, fileDescriptor.rightsBase, true);
				view.setBigUint64(fdstat + 16, fileDescriptor.rightsInheriting, true);
				return 0;
			});
			this.fd_fdstat_set_flags = syscallWrap(this, "fd_fdstat_set_flags", function(_fd, _flags) {
				return 52;
			});
			this.fd_fdstat_set_rights = syscallWrap(this, "fd_fdstat_set_rights", function(fd, rightsBase, rightsInheriting) {
				const fileDescriptor = _wasi.get(this).fds.get(fd, BigInt(0), BigInt(0));
				if ((rightsBase | fileDescriptor.rightsBase) > fileDescriptor.rightsBase) return 76;
				if ((rightsInheriting | fileDescriptor.rightsInheriting) > fileDescriptor.rightsInheriting) return 76;
				fileDescriptor.rightsBase = rightsBase;
				fileDescriptor.rightsInheriting = rightsInheriting;
				return 0;
			});
			this.fd_prestat_get = syscallWrap(this, "fd_prestat_get", function(fd, prestat) {
				prestat = Number(prestat);
				if (prestat === 0) return 28;
				const wasi = _wasi.get(this);
				let fileDescriptor;
				try {
					fileDescriptor = wasi.fds.get(fd, BigInt(0), BigInt(0));
				} catch (err) {
					if (err instanceof WasiError) return err.errno;
					throw err;
				}
				if (fileDescriptor.preopen !== 1) return 28;
				const { view } = getMemory(this);
				view.setUint32(prestat, 0, true);
				view.setUint32(prestat + 4, encoder.encode(fileDescriptor.path).length, true);
				return 0;
			});
			this.fd_prestat_dir_name = syscallWrap(this, "fd_prestat_dir_name", function(fd, path, path_len) {
				path = Number(path);
				path_len = Number(path_len);
				if (path === 0) return 28;
				const fileDescriptor = _wasi.get(this).fds.get(fd, BigInt(0), BigInt(0));
				if (fileDescriptor.preopen !== 1) return 8;
				const buffer = encoder.encode(fileDescriptor.path);
				if (buffer.length > path_len) return 42;
				const { HEAPU8 } = getMemory(this);
				HEAPU8.set(buffer, path);
				return 0;
			});
			this.fd_seek = syscallWrap(this, "fd_seek", function(fd, offset, whence, newOffset) {
				newOffset = Number(newOffset);
				if (newOffset === 0) return 28;
				if (fd === 0 || fd === 1 || fd === 2) return 0;
				const r = _wasi.get(this).fds.get(fd, WasiRights.FD_SEEK, BigInt(0)).seek(offset, whence);
				const { view } = getMemory(this);
				view.setBigUint64(newOffset, r, true);
				return 0;
			});
			this.fd_tell = syscallWrap(this, "fd_tell", function(fd, offset) {
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_TELL, BigInt(0));
				const pos = BigInt(fileDescriptor.pos);
				const { view } = getMemory(this);
				view.setBigUint64(Number(offset), pos, true);
				return 0;
			});
			this.poll_oneoff = syscallWrap(this, "poll_oneoff", function(in_ptr, out_ptr, nsubscriptions, nevents) {
				in_ptr = Number(in_ptr);
				out_ptr = Number(out_ptr);
				nevents = Number(nevents);
				nsubscriptions = Number(nsubscriptions);
				nsubscriptions = nsubscriptions >>> 0;
				if (in_ptr === 0 || out_ptr === 0 || nsubscriptions === 0 || nevents === 0) return 28;
				const { view } = getMemory(this);
				view.setUint32(nevents, 0, true);
				let i = 0;
				let timer_userdata = BigInt(0);
				let cur_timeout = BigInt(0);
				let has_timeout = 0;
				let min_timeout = BigInt(0);
				let sub;
				const subscriptions = Array(nsubscriptions);
				for (i = 0; i < nsubscriptions; i++) {
					sub = in_ptr + i * 48;
					const userdata = view.getBigUint64(sub, true);
					const type = view.getUint8(sub + 8);
					const clockIdOrFd = view.getUint32(sub + 16, true);
					const timeout = view.getBigUint64(sub + 24, true);
					const precision = view.getBigUint64(sub + 32, true);
					const flags = view.getUint16(sub + 40, true);
					subscriptions[i] = {
						userdata,
						type,
						u: {
							clock: {
								clock_id: clockIdOrFd,
								timeout,
								precision,
								flags
							},
							fd_readwrite: { fd: clockIdOrFd }
						}
					};
				}
				const fdevents = [];
				for (i = 0; i < nsubscriptions; i++) {
					sub = subscriptions[i];
					switch (sub.type) {
						case 0:
							if (sub.u.clock.flags === 1) {
								const now = BigInt(Date.now()) * BigInt(1e6);
								cur_timeout = sub.u.clock.timeout - now;
							} else cur_timeout = sub.u.clock.timeout;
							if (has_timeout === 0 || cur_timeout < min_timeout) {
								min_timeout = cur_timeout;
								timer_userdata = sub.userdata;
								has_timeout = 1;
							}
							break;
						case 1:
						case 2:
							fdevents.push(sub);
							break;
						default: return 28;
					}
				}
				if (fdevents.length > 0) {
					for (i = 0; i < fdevents.length; i++) {
						const fdevent = fdevents[i];
						const event = out_ptr + 32 * i;
						view.setBigUint64(event, fdevent.userdata, true);
						view.setUint32(event + 8, 52, true);
						view.setUint32(event + 12, fdevent.type, true);
						view.setBigUint64(event + 16, BigInt(0), true);
						view.setUint16(event + 24, 0, true);
						view.setUint32(nevents, 1, true);
					}
					view.setUint32(nevents, fdevents.length, true);
					return 0;
				}
				if (has_timeout) {
					const delay = Number(min_timeout / BigInt(1e6));
					sleepBreakIf(delay, () => false);
					const event = out_ptr;
					view.setBigUint64(event, timer_userdata, true);
					view.setUint32(event + 8, 0, true);
					view.setUint32(event + 12, 0, true);
					view.setUint32(nevents, 1, true);
				}
				return 0;
			});
			this.proc_exit = syscallWrap(this, "proc_exit", function(rval) {
				if (typeof process === "object" && process !== null && typeof process.exit === "function") process.exit(rval);
				return 0;
			});
			this.proc_raise = syscallWrap(this, "proc_raise", function(_sig) {
				return 52;
			});
			this.sched_yield = syscallWrap(this, "sched_yield", function() {
				return 0;
			});
			this.random_get = typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function" ? syscallWrap(this, "random_get", function(buf, buf_len) {
				buf = Number(buf);
				if (buf === 0) return 28;
				buf_len = Number(buf_len);
				const { HEAPU8, view } = getMemory(this);
				if (typeof SharedArrayBuffer === "function" && HEAPU8.buffer instanceof SharedArrayBuffer || Object.prototype.toString.call(HEAPU8.buffer) === "[object SharedArrayBuffer]") {
					for (let i = buf; i < buf + buf_len; ++i) view.setUint8(i, Math.floor(Math.random() * 256));
					return 0;
				}
				let pos;
				const stride = 65536;
				for (pos = 0; pos + stride < buf_len; pos += stride) crypto.getRandomValues(HEAPU8.subarray(buf + pos, buf + pos + stride));
				crypto.getRandomValues(HEAPU8.subarray(buf + pos, buf + buf_len));
				return 0;
			}) : syscallWrap(this, "random_get", function(buf, buf_len) {
				buf = Number(buf);
				if (buf === 0) return 28;
				buf_len = Number(buf_len);
				const { view } = getMemory(this);
				for (let i = buf; i < buf + buf_len; ++i) view.setUint8(i, Math.floor(Math.random() * 256));
				return 0;
			});
			this.sock_recv = syscallWrap(this, "sock_recv", function() {
				return 58;
			});
			this.sock_send = syscallWrap(this, "sock_send", function() {
				return 58;
			});
			this.sock_shutdown = syscallWrap(this, "sock_shutdown", function() {
				return 58;
			});
			this.sock_accept = syscallWrap(this, "sock_accept", function() {
				return 58;
			});
			_wasi.set(this, {
				fds,
				args,
				env
			});
			if (fs$1) _fs.set(this, fs$1);
			const _this = this;
			function defineImport(name, syncVersion, asyncVersion, parameterType, returnType) {
				if (asyncFs) if (asyncify) _this[name] = asyncify.wrapImportFunction(syscallWrap(_this, name, asyncVersion));
				else _this[name] = wrapAsyncImport(syscallWrap(_this, name, asyncVersion), parameterType, returnType);
				else _this[name] = syscallWrap(_this, name, syncVersion);
			}
			defineImport("fd_allocate", function fd_allocate(fd, offset, len) {
				const wasi = _wasi.get(this);
				const fs$2 = getFs(this);
				const fileDescriptor = wasi.fds.get(fd, WasiRights.FD_ALLOCATE, BigInt(0));
				if (fs$2.fstatSync(fileDescriptor.fd, { bigint: true }).size < offset + len) fs$2.ftruncateSync(fileDescriptor.fd, Number(offset + len));
				return 0;
			}, async function fd_allocate(fd, offset, len) {
				const h = _wasi.get(this).fds.get(fd, WasiRights.FD_ALLOCATE, BigInt(0)).fd;
				if ((await h.stat({ bigint: true })).size < offset + len) await h.truncate(Number(offset + len));
				return 0;
			}, [
				"i32",
				"i64",
				"f64"
			], ["i32"]);
			defineImport("fd_close", function fd_close(fd) {
				const wasi = _wasi.get(this);
				const fileDescriptor = wasi.fds.get(fd, BigInt(0), BigInt(0));
				getFs(this).closeSync(fileDescriptor.fd);
				wasi.fds.remove(fd);
				return 0;
			}, async function fd_close(fd) {
				const wasi = _wasi.get(this);
				await wasi.fds.get(fd, BigInt(0), BigInt(0)).fd.close();
				wasi.fds.remove(fd);
				return 0;
			}, ["i32"], ["i32"]);
			defineImport("fd_datasync", function fd_datasync(fd) {
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_DATASYNC, BigInt(0));
				getFs(this).fdatasyncSync(fileDescriptor.fd);
				return 0;
			}, async function fd_datasync(fd) {
				await _wasi.get(this).fds.get(fd, WasiRights.FD_DATASYNC, BigInt(0)).fd.datasync();
				return 0;
			}, ["i32"], ["i32"]);
			defineImport("fd_filestat_get", function fd_filestat_get(fd, buf) {
				buf = Number(buf);
				if (buf === 0) return 28;
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_FILESTAT_GET, BigInt(0));
				const stat = getFs(this).fstatSync(fileDescriptor.fd, { bigint: true });
				const { view } = getMemory(this);
				toFileStat(view, buf, stat);
				return 0;
			}, async function fd_filestat_get(fd, buf) {
				buf = Number(buf);
				if (buf === 0) return 28;
				const stat = await _wasi.get(this).fds.get(fd, WasiRights.FD_FILESTAT_GET, BigInt(0)).fd.stat({ bigint: true });
				const { view } = getMemory(this);
				toFileStat(view, buf, stat);
				return 0;
			}, ["i32", "i32"], ["i32"]);
			defineImport("fd_filestat_set_size", function fd_filestat_set_size(fd, size) {
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_FILESTAT_SET_SIZE, BigInt(0));
				getFs(this).ftruncateSync(fileDescriptor.fd, Number(size));
				return 0;
			}, async function fd_filestat_set_size(fd, size) {
				await _wasi.get(this).fds.get(fd, WasiRights.FD_FILESTAT_SET_SIZE, BigInt(0)).fd.truncate(Number(size));
				return 0;
			}, ["i32", "i64"], ["i32"]);
			function fdFilestatGetTimes(fd, atim, mtim, flags) {
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_FILESTAT_SET_TIMES, BigInt(0));
				if ((flags & 2) === 2) atim = BigInt(Date.now() * 1e6);
				if ((flags & 8) === 8) mtim = BigInt(Date.now() * 1e6);
				return {
					fileDescriptor,
					atim,
					mtim
				};
			}
			defineImport("fd_filestat_set_times", function fd_filestat_set_times(fd, atim, mtim, flags) {
				if (validateFstFlagsOrReturn(flags)) return 28;
				const { fileDescriptor, atim: atimRes, mtim: mtimRes } = fdFilestatGetTimes.call(this, fd, atim, mtim, flags);
				getFs(this).futimesSync(fileDescriptor.fd, Number(atimRes), Number(mtimRes));
				return 0;
			}, async function fd_filestat_set_times(fd, atim, mtim, flags) {
				if (validateFstFlagsOrReturn(flags)) return 28;
				const { fileDescriptor, atim: atimRes, mtim: mtimRes } = fdFilestatGetTimes.call(this, fd, atim, mtim, flags);
				await fileDescriptor.fd.utimes(Number(atimRes), Number(mtimRes));
				return 0;
			}, [
				"i32",
				"i64",
				"i64",
				"i32"
			], ["i32"]);
			defineImport("fd_pread", function fd_pread(fd, iovs, iovslen, offset, size) {
				iovs = Number(iovs);
				size = Number(size);
				if (iovs === 0 && iovslen || size === 0 || offset > INT64_MAX) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_READ | WasiRights.FD_SEEK, BigInt(0));
				if (!iovslen) {
					view.setUint32(size, 0, true);
					return 0;
				}
				let totalSize = 0;
				const ioVecs = Array.from({ length: Number(iovslen) }, (_, i) => {
					const offset$1 = iovs + i * 8;
					const buf = view.getInt32(offset$1, true);
					const bufLen = view.getUint32(offset$1 + 4, true);
					totalSize += bufLen;
					return HEAPU8.subarray(buf, buf + bufLen);
				});
				let nread = 0;
				const buffer = (() => {
					try {
						return new Uint8Array(new SharedArrayBuffer(totalSize));
					} catch (_) {
						return new Uint8Array(totalSize);
					}
				})();
				buffer._isBuffer = true;
				const bytesRead = getFs(this).readSync(fileDescriptor.fd, buffer, 0, buffer.length, Number(offset));
				nread = buffer ? copyMemory(ioVecs, buffer.subarray(0, bytesRead)) : 0;
				view.setUint32(size, nread, true);
				return 0;
			}, async function(fd, iovs, iovslen, offset, size) {
				iovs = Number(iovs);
				size = Number(size);
				if (iovs === 0 && iovslen || size === 0 || offset > INT64_MAX) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_READ | WasiRights.FD_SEEK, BigInt(0));
				if (!iovslen) {
					view.setUint32(size, 0, true);
					return 0;
				}
				let totalSize = 0;
				const ioVecs = Array.from({ length: Number(iovslen) }, (_, i) => {
					const offset$1 = iovs + i * 8;
					const buf = view.getInt32(offset$1, true);
					const bufLen = view.getUint32(offset$1 + 4, true);
					totalSize += bufLen;
					return HEAPU8.subarray(buf, buf + bufLen);
				});
				let nread = 0;
				const buffer = new Uint8Array(totalSize);
				buffer._isBuffer = true;
				const { bytesRead } = await fileDescriptor.fd.read(buffer, 0, buffer.length, Number(offset));
				nread = buffer ? copyMemory(ioVecs, buffer.subarray(0, bytesRead)) : 0;
				view.setUint32(size, nread, true);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i64",
				"i32"
			], ["i32"]);
			defineImport("fd_pwrite", function fd_pwrite(fd, iovs, iovslen, offset, size) {
				iovs = Number(iovs);
				size = Number(size);
				if (iovs === 0 && iovslen || size === 0 || offset > INT64_MAX) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_WRITE | WasiRights.FD_SEEK, BigInt(0));
				if (!iovslen) {
					view.setUint32(size, 0, true);
					return 0;
				}
				const buffer = concatBuffer(Array.from({ length: Number(iovslen) }, (_, i) => {
					const offset$1 = iovs + i * 8;
					const buf = view.getInt32(offset$1, true);
					const bufLen = view.getUint32(offset$1 + 4, true);
					return HEAPU8.subarray(buf, buf + bufLen);
				}));
				const nwritten = getFs(this).writeSync(fileDescriptor.fd, buffer, 0, buffer.length, Number(offset));
				view.setUint32(size, nwritten, true);
				return 0;
			}, async function fd_pwrite(fd, iovs, iovslen, offset, size) {
				iovs = Number(iovs);
				size = Number(size);
				if (iovs === 0 && iovslen || size === 0 || offset > INT64_MAX) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_WRITE | WasiRights.FD_SEEK, BigInt(0));
				if (!iovslen) {
					view.setUint32(size, 0, true);
					return 0;
				}
				const buffer = concatBuffer(Array.from({ length: Number(iovslen) }, (_, i) => {
					const offset$1 = iovs + i * 8;
					const buf = view.getInt32(offset$1, true);
					const bufLen = view.getUint32(offset$1 + 4, true);
					return HEAPU8.subarray(buf, buf + bufLen);
				}));
				const { bytesWritten } = await fileDescriptor.fd.write(buffer, 0, buffer.length, Number(offset));
				view.setUint32(size, bytesWritten, true);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i64",
				"i32"
			], ["i32"]);
			defineImport("fd_read", function fd_read(fd, iovs, iovslen, size) {
				iovs = Number(iovs);
				size = Number(size);
				if (iovs === 0 && iovslen || size === 0) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_READ, BigInt(0));
				if (!iovslen) {
					view.setUint32(size, 0, true);
					return 0;
				}
				let totalSize = 0;
				const ioVecs = Array.from({ length: Number(iovslen) }, (_, i) => {
					const offset = iovs + i * 8;
					const buf = view.getInt32(offset, true);
					const bufLen = view.getUint32(offset + 4, true);
					totalSize += bufLen;
					return HEAPU8.subarray(buf, buf + bufLen);
				});
				let buffer;
				let nread = 0;
				if (fd === 0) {
					if (typeof window === "undefined" || typeof window.prompt !== "function") return 58;
					buffer = readStdin();
					nread = buffer ? copyMemory(ioVecs, buffer) : 0;
				} else {
					buffer = (() => {
						try {
							return new Uint8Array(new SharedArrayBuffer(totalSize));
						} catch (_) {
							return new Uint8Array(totalSize);
						}
					})();
					buffer._isBuffer = true;
					const bytesRead = getFs(this).readSync(fileDescriptor.fd, buffer, 0, buffer.length, Number(fileDescriptor.pos));
					nread = buffer ? copyMemory(ioVecs, buffer.subarray(0, bytesRead)) : 0;
					fileDescriptor.pos += BigInt(nread);
				}
				view.setUint32(size, nread, true);
				return 0;
			}, async function fd_read(fd, iovs, iovslen, size) {
				iovs = Number(iovs);
				size = Number(size);
				if (iovs === 0 && iovslen || size === 0) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_READ, BigInt(0));
				if (!iovslen) {
					view.setUint32(size, 0, true);
					return 0;
				}
				let totalSize = 0;
				const ioVecs = Array.from({ length: Number(iovslen) }, (_, i) => {
					const offset = iovs + i * 8;
					const buf = view.getInt32(offset, true);
					const bufLen = view.getUint32(offset + 4, true);
					totalSize += bufLen;
					return HEAPU8.subarray(buf, buf + bufLen);
				});
				let buffer;
				let nread = 0;
				if (fd === 0) {
					if (typeof window === "undefined" || typeof window.prompt !== "function") return 58;
					buffer = readStdin();
					nread = buffer ? copyMemory(ioVecs, buffer) : 0;
				} else {
					buffer = new Uint8Array(totalSize);
					buffer._isBuffer = true;
					const { bytesRead } = await fileDescriptor.fd.read(buffer, 0, buffer.length, Number(fileDescriptor.pos));
					nread = buffer ? copyMemory(ioVecs, buffer.subarray(0, bytesRead)) : 0;
					fileDescriptor.pos += BigInt(nread);
				}
				view.setUint32(size, nread, true);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i32"
			], ["i32"]);
			defineImport("fd_readdir", function fd_readdir(fd, buf, buf_len, cookie, bufused) {
				buf = Number(buf);
				buf_len = Number(buf_len);
				bufused = Number(bufused);
				if (buf === 0 || bufused === 0) return 0;
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_READDIR, BigInt(0));
				const fs$2 = getFs(this);
				const entries = fs$2.readdirSync(fileDescriptor.realPath, { withFileTypes: true });
				const { HEAPU8, view } = getMemory(this);
				let bufferUsed = 0;
				for (let i = Number(cookie); i < entries.length; i++) {
					const nameData = encoder.encode(entries[i].name);
					const entryInfo = fs$2.statSync(resolve(fileDescriptor.realPath, entries[i].name), { bigint: true });
					const entryData = new Uint8Array(24 + nameData.byteLength);
					const entryView = new DataView(entryData.buffer);
					entryView.setBigUint64(0, BigInt(i + 1), true);
					entryView.setBigUint64(8, BigInt(entryInfo.ino ? entryInfo.ino : 0), true);
					entryView.setUint32(16, nameData.byteLength, true);
					let type;
					if (entries[i].isFile()) type = 4;
					else if (entries[i].isDirectory()) type = 3;
					else if (entries[i].isSymbolicLink()) type = 7;
					else if (entries[i].isCharacterDevice()) type = 2;
					else if (entries[i].isBlockDevice()) type = 1;
					else if (entries[i].isSocket()) type = 6;
					else type = 0;
					entryView.setUint8(20, type);
					entryData.set(nameData, 24);
					const data = entryData.slice(0, Math.min(entryData.length, buf_len - bufferUsed));
					HEAPU8.set(data, buf + bufferUsed);
					bufferUsed += data.byteLength;
				}
				view.setUint32(bufused, bufferUsed, true);
				return 0;
			}, async function fd_readdir(fd, buf, buf_len, cookie, bufused) {
				buf = Number(buf);
				buf_len = Number(buf_len);
				bufused = Number(bufused);
				if (buf === 0 || bufused === 0) return 0;
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_READDIR, BigInt(0));
				const fs$2 = getFs(this);
				const entries = await fs$2.promises.readdir(fileDescriptor.realPath, { withFileTypes: true });
				const { HEAPU8, view } = getMemory(this);
				let bufferUsed = 0;
				for (let i = Number(cookie); i < entries.length; i++) {
					const nameData = encoder.encode(entries[i].name);
					const entryInfo = await fs$2.promises.stat(resolve(fileDescriptor.realPath, entries[i].name), { bigint: true });
					const entryData = new Uint8Array(24 + nameData.byteLength);
					const entryView = new DataView(entryData.buffer);
					entryView.setBigUint64(0, BigInt(i + 1), true);
					entryView.setBigUint64(8, BigInt(entryInfo.ino ? entryInfo.ino : 0), true);
					entryView.setUint32(16, nameData.byteLength, true);
					let type;
					if (entries[i].isFile()) type = 4;
					else if (entries[i].isDirectory()) type = 3;
					else if (entries[i].isSymbolicLink()) type = 7;
					else if (entries[i].isCharacterDevice()) type = 2;
					else if (entries[i].isBlockDevice()) type = 1;
					else if (entries[i].isSocket()) type = 6;
					else type = 0;
					entryView.setUint8(20, type);
					entryData.set(nameData, 24);
					const data = entryData.slice(0, Math.min(entryData.length, buf_len - bufferUsed));
					HEAPU8.set(data, buf + bufferUsed);
					bufferUsed += data.byteLength;
				}
				view.setUint32(bufused, bufferUsed, true);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i64",
				"i32"
			], ["i32"]);
			defineImport("fd_renumber", function fd_renumber(from, to) {
				_wasi.get(this).fds.renumber(to, from);
				return 0;
			}, async function fd_renumber(from, to) {
				await _wasi.get(this).fds.renumber(to, from);
				return 0;
			}, ["i32", "i32"], ["i32"]);
			defineImport("fd_sync", function fd_sync(fd) {
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_SYNC, BigInt(0));
				getFs(this).fsyncSync(fileDescriptor.fd);
				return 0;
			}, async function fd_sync(fd) {
				await _wasi.get(this).fds.get(fd, WasiRights.FD_SYNC, BigInt(0)).fd.sync();
				return 0;
			}, ["i32"], ["i32"]);
			defineImport("fd_write", function fd_write(fd, iovs, iovslen, size) {
				iovs = Number(iovs);
				size = Number(size);
				if (iovs === 0 && iovslen || size === 0) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_WRITE, BigInt(0));
				if (!iovslen) {
					view.setUint32(size, 0, true);
					return 0;
				}
				const buffer = concatBuffer(Array.from({ length: Number(iovslen) }, (_, i) => {
					const offset = iovs + i * 8;
					const buf = view.getInt32(offset, true);
					const bufLen = view.getUint32(offset + 4, true);
					return HEAPU8.subarray(buf, buf + bufLen);
				}));
				let nwritten;
				if (fd === 1 || fd === 2) nwritten = fileDescriptor.write(buffer);
				else {
					nwritten = getFs(this).writeSync(fileDescriptor.fd, buffer, 0, buffer.length, Number(fileDescriptor.pos));
					fileDescriptor.pos += BigInt(nwritten);
				}
				view.setUint32(size, nwritten, true);
				return 0;
			}, async function fd_write(fd, iovs, iovslen, size) {
				iovs = Number(iovs);
				size = Number(size);
				if (iovs === 0 && iovslen || size === 0) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.FD_WRITE, BigInt(0));
				if (!iovslen) {
					view.setUint32(size, 0, true);
					return 0;
				}
				const buffer = concatBuffer(Array.from({ length: Number(iovslen) }, (_, i) => {
					const offset = iovs + i * 8;
					const buf = view.getInt32(offset, true);
					const bufLen = view.getUint32(offset + 4, true);
					return HEAPU8.subarray(buf, buf + bufLen);
				}));
				let nwritten;
				if (fd === 1 || fd === 2) nwritten = fileDescriptor.write(buffer);
				else {
					nwritten = await (await fileDescriptor.fd.write(buffer, 0, buffer.length, Number(fileDescriptor.pos))).bytesWritten;
					fileDescriptor.pos += BigInt(nwritten);
				}
				view.setUint32(size, nwritten, true);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i32"
			], ["i32"]);
			defineImport("path_create_directory", function path_create_directory(fd, path, path_len) {
				path = Number(path);
				path_len = Number(path_len);
				if (path === 0) return 28;
				const { HEAPU8 } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_CREATE_DIRECTORY, BigInt(0));
				let pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				pathString = resolve(fileDescriptor.realPath, pathString);
				getFs(this).mkdirSync(pathString);
				return 0;
			}, async function path_create_directory(fd, path, path_len) {
				path = Number(path);
				path_len = Number(path_len);
				if (path === 0) return 28;
				const { HEAPU8 } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_CREATE_DIRECTORY, BigInt(0));
				let pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				pathString = resolve(fileDescriptor.realPath, pathString);
				await getFs(this).promises.mkdir(pathString);
				return 0;
			}, [
				"i32",
				"i32",
				"i32"
			], ["i32"]);
			defineImport("path_filestat_get", function path_filestat_get(fd, flags, path, path_len, filestat) {
				path = Number(path);
				path_len = Number(path_len);
				filestat = Number(filestat);
				if (path === 0 || filestat === 0) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_FILESTAT_GET, BigInt(0));
				let pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				const fs$2 = getFs(this);
				pathString = resolve(fileDescriptor.realPath, pathString);
				let stat;
				if ((flags & 1) === 1) stat = fs$2.statSync(pathString, { bigint: true });
				else stat = fs$2.lstatSync(pathString, { bigint: true });
				toFileStat(view, filestat, stat);
				return 0;
			}, async function path_filestat_get(fd, flags, path, path_len, filestat) {
				path = Number(path);
				path_len = Number(path_len);
				filestat = Number(filestat);
				if (path === 0 || filestat === 0) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_FILESTAT_GET, BigInt(0));
				let pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				const fs$2 = getFs(this);
				pathString = resolve(fileDescriptor.realPath, pathString);
				let stat;
				if ((flags & 1) === 1) stat = await fs$2.promises.stat(pathString, { bigint: true });
				else stat = await fs$2.promises.lstat(pathString, { bigint: true });
				toFileStat(view, filestat, stat);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i32",
				"i32"
			], ["i32"]);
			defineImport("path_filestat_set_times", function path_filestat_set_times(fd, flags, path, path_len, atim, mtim, fst_flags) {
				path = Number(path);
				path_len = Number(path_len);
				if (path === 0) return 28;
				const { HEAPU8 } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_FILESTAT_SET_TIMES, BigInt(0));
				if (validateFstFlagsOrReturn(fst_flags)) return 28;
				const fs$2 = getFs(this);
				const resolvedPath = resolvePathSync(fs$2, fileDescriptor, decoder.decode(unsharedSlice(HEAPU8, path, path + path_len)), flags);
				if ((fst_flags & 2) === 2) atim = BigInt(Date.now() * 1e6);
				if ((fst_flags & 8) === 8) mtim = BigInt(Date.now() * 1e6);
				fs$2.utimesSync(resolvedPath, Number(atim), Number(mtim));
				return 0;
			}, async function path_filestat_set_times(fd, flags, path, path_len, atim, mtim, fst_flags) {
				path = Number(path);
				path_len = Number(path_len);
				if (path === 0) return 28;
				const { HEAPU8 } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_FILESTAT_SET_TIMES, BigInt(0));
				if (validateFstFlagsOrReturn(fst_flags)) return 28;
				const fs$2 = getFs(this);
				const resolvedPath = await resolvePathAsync(fs$2, fileDescriptor, decoder.decode(unsharedSlice(HEAPU8, path, path + path_len)), flags);
				if ((fst_flags & 2) === 2) atim = BigInt(Date.now() * 1e6);
				if ((fst_flags & 8) === 8) mtim = BigInt(Date.now() * 1e6);
				await fs$2.promises.utimes(resolvedPath, Number(atim), Number(mtim));
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i32",
				"i64",
				"i64",
				"i32"
			], ["i32"]);
			defineImport("path_link", function path_link(old_fd, old_flags, old_path, old_path_len, new_fd, new_path, new_path_len) {
				old_path = Number(old_path);
				old_path_len = Number(old_path_len);
				new_path = Number(new_path);
				new_path_len = Number(new_path_len);
				if (old_path === 0 || new_path === 0) return 28;
				const wasi = _wasi.get(this);
				let oldWrap;
				let newWrap;
				if (old_fd === new_fd) oldWrap = newWrap = wasi.fds.get(old_fd, WasiRights.PATH_LINK_SOURCE | WasiRights.PATH_LINK_TARGET, BigInt(0));
				else {
					oldWrap = wasi.fds.get(old_fd, WasiRights.PATH_LINK_SOURCE, BigInt(0));
					newWrap = wasi.fds.get(new_fd, WasiRights.PATH_LINK_TARGET, BigInt(0));
				}
				const { HEAPU8 } = getMemory(this);
				const fs$2 = getFs(this);
				const resolvedOldPath = resolvePathSync(fs$2, oldWrap, decoder.decode(unsharedSlice(HEAPU8, old_path, old_path + old_path_len)), old_flags);
				const resolvedNewPath = resolve(newWrap.realPath, decoder.decode(unsharedSlice(HEAPU8, new_path, new_path + new_path_len)));
				fs$2.linkSync(resolvedOldPath, resolvedNewPath);
				return 0;
			}, async function path_link(old_fd, old_flags, old_path, old_path_len, new_fd, new_path, new_path_len) {
				old_path = Number(old_path);
				old_path_len = Number(old_path_len);
				new_path = Number(new_path);
				new_path_len = Number(new_path_len);
				if (old_path === 0 || new_path === 0) return 28;
				const wasi = _wasi.get(this);
				let oldWrap;
				let newWrap;
				if (old_fd === new_fd) oldWrap = newWrap = wasi.fds.get(old_fd, WasiRights.PATH_LINK_SOURCE | WasiRights.PATH_LINK_TARGET, BigInt(0));
				else {
					oldWrap = wasi.fds.get(old_fd, WasiRights.PATH_LINK_SOURCE, BigInt(0));
					newWrap = wasi.fds.get(new_fd, WasiRights.PATH_LINK_TARGET, BigInt(0));
				}
				const { HEAPU8 } = getMemory(this);
				const fs$2 = getFs(this);
				const resolvedOldPath = await resolvePathAsync(fs$2, oldWrap, decoder.decode(unsharedSlice(HEAPU8, old_path, old_path + old_path_len)), old_flags);
				const resolvedNewPath = resolve(newWrap.realPath, decoder.decode(unsharedSlice(HEAPU8, new_path, new_path + new_path_len)));
				await fs$2.promises.link(resolvedOldPath, resolvedNewPath);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i32",
				"i32",
				"i32",
				"i32"
			], ["i32"]);
			function pathOpen(o_flags, fs_rights_base, fs_rights_inheriting, fs_flags) {
				const read = (fs_rights_base & (WasiRights.FD_READ | WasiRights.FD_READDIR)) !== BigInt(0);
				const write = (fs_rights_base & (WasiRights.FD_DATASYNC | WasiRights.FD_WRITE | WasiRights.FD_ALLOCATE | WasiRights.FD_FILESTAT_SET_SIZE)) !== BigInt(0);
				let flags = write ? read ? 2 : 1 : 0;
				let needed_base = WasiRights.PATH_OPEN;
				let needed_inheriting = fs_rights_base | fs_rights_inheriting;
				if ((o_flags & 1) !== 0) {
					flags |= 64;
					needed_base |= WasiRights.PATH_CREATE_FILE;
				}
				if ((o_flags & 2) !== 0) flags |= 65536;
				if ((o_flags & 4) !== 0) flags |= 128;
				if ((o_flags & 8) !== 0) {
					flags |= 512;
					needed_base |= WasiRights.PATH_FILESTAT_SET_SIZE;
				}
				if ((fs_flags & 1) !== 0) flags |= 1024;
				if ((fs_flags & 2) !== 0) needed_inheriting |= WasiRights.FD_DATASYNC;
				if ((fs_flags & 4) !== 0) flags |= 2048;
				if ((fs_flags & 8) !== 0) {
					flags |= 1052672;
					needed_inheriting |= WasiRights.FD_SYNC;
				}
				if ((fs_flags & 16) !== 0) {
					flags |= 1052672;
					needed_inheriting |= WasiRights.FD_SYNC;
				}
				if (write && (flags & 1536) === 0) needed_inheriting |= WasiRights.FD_SEEK;
				return {
					flags,
					needed_base,
					needed_inheriting
				};
			}
			defineImport("path_open", function path_open(dirfd, dirflags, path, path_len, o_flags, fs_rights_base, fs_rights_inheriting, fs_flags, fd) {
				path = Number(path);
				fd = Number(fd);
				if (path === 0 || fd === 0) return 28;
				path_len = Number(path_len);
				fs_rights_base = BigInt(fs_rights_base);
				fs_rights_inheriting = BigInt(fs_rights_inheriting);
				const { flags: flagsRes, needed_base: neededBase, needed_inheriting: neededInheriting } = pathOpen(o_flags, fs_rights_base, fs_rights_inheriting, fs_flags);
				const wasi = _wasi.get(this);
				const fileDescriptor = wasi.fds.get(dirfd, neededBase, neededInheriting);
				const memory = getMemory(this);
				const HEAPU8 = memory.HEAPU8;
				const pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				const fs$2 = getFs(this);
				const resolved_path = resolvePathSync(fs$2, fileDescriptor, pathString, dirflags);
				const r = fs$2.openSync(resolved_path, flagsRes, 438);
				const filetype = wasi.fds.getFileTypeByFd(r);
				if (filetype !== 3 && ((o_flags & 2) !== 0 || resolved_path.endsWith("/"))) return 54;
				const { base: max_base, inheriting: max_inheriting } = getRights(wasi.fds.stdio, r, flagsRes, filetype);
				const wrap$2 = wasi.fds.insert(r, resolved_path, resolved_path, filetype, fs_rights_base & max_base, fs_rights_inheriting & max_inheriting, 0);
				const stat = fs$2.fstatSync(r, { bigint: true });
				if (stat.isFile()) {
					wrap$2.size = stat.size;
					if ((flagsRes & 1024) !== 0) wrap$2.pos = stat.size;
				}
				memory.view.setInt32(fd, wrap$2.id, true);
				return 0;
			}, async function path_open(dirfd, dirflags, path, path_len, o_flags, fs_rights_base, fs_rights_inheriting, fs_flags, fd) {
				path = Number(path);
				fd = Number(fd);
				if (path === 0 || fd === 0) return 28;
				path_len = Number(path_len);
				fs_rights_base = BigInt(fs_rights_base);
				fs_rights_inheriting = BigInt(fs_rights_inheriting);
				const { flags: flagsRes, needed_base: neededBase, needed_inheriting: neededInheriting } = pathOpen(o_flags, fs_rights_base, fs_rights_inheriting, fs_flags);
				const wasi = _wasi.get(this);
				const fileDescriptor = wasi.fds.get(dirfd, neededBase, neededInheriting);
				const memory = getMemory(this);
				const HEAPU8 = memory.HEAPU8;
				const pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				const fs$2 = getFs(this);
				const resolved_path = await resolvePathAsync(fs$2, fileDescriptor, pathString, dirflags);
				const r = await fs$2.promises.open(resolved_path, flagsRes, 438);
				const filetype = await wasi.fds.getFileTypeByFd(r);
				if ((o_flags & 2) !== 0 && filetype !== 3) return 54;
				const { base: max_base, inheriting: max_inheriting } = getRights(wasi.fds.stdio, r.fd, flagsRes, filetype);
				const wrap$2 = wasi.fds.insert(r, resolved_path, resolved_path, filetype, fs_rights_base & max_base, fs_rights_inheriting & max_inheriting, 0);
				const stat = await r.stat({ bigint: true });
				if (stat.isFile()) {
					wrap$2.size = stat.size;
					if ((flagsRes & 1024) !== 0) wrap$2.pos = stat.size;
				}
				memory.view.setInt32(fd, wrap$2.id, true);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i32",
				"i32",
				"i64",
				"i64",
				"i32",
				"i32"
			], ["i32"]);
			defineImport("path_readlink", function path_readlink(fd, path, path_len, buf, buf_len, bufused) {
				path = Number(path);
				path_len = Number(path_len);
				buf = Number(buf);
				buf_len = Number(buf_len);
				bufused = Number(bufused);
				if (path === 0 || buf === 0 || bufused === 0) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_READLINK, BigInt(0));
				let pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				pathString = resolve(fileDescriptor.realPath, pathString);
				const link = getFs(this).readlinkSync(pathString);
				const linkData = encoder.encode(link);
				const len = Math.min(linkData.length, buf_len);
				if (len >= buf_len) return 42;
				HEAPU8.set(linkData.subarray(0, len), buf);
				HEAPU8[buf + len] = 0;
				view.setUint32(bufused, len, true);
				return 0;
			}, async function path_readlink(fd, path, path_len, buf, buf_len, bufused) {
				path = Number(path);
				path_len = Number(path_len);
				buf = Number(buf);
				buf_len = Number(buf_len);
				bufused = Number(bufused);
				if (path === 0 || buf === 0 || bufused === 0) return 28;
				const { HEAPU8, view } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_READLINK, BigInt(0));
				let pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				pathString = resolve(fileDescriptor.realPath, pathString);
				const link = await getFs(this).promises.readlink(pathString);
				const linkData = encoder.encode(link);
				const len = Math.min(linkData.length, buf_len);
				if (len >= buf_len) return 42;
				HEAPU8.set(linkData.subarray(0, len), buf);
				HEAPU8[buf + len] = 0;
				view.setUint32(bufused, len, true);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i32",
				"i32",
				"i32"
			], ["i32"]);
			defineImport("path_remove_directory", function path_remove_directory(fd, path, path_len) {
				path = Number(path);
				path_len = Number(path_len);
				if (path === 0) return 28;
				const { HEAPU8 } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_REMOVE_DIRECTORY, BigInt(0));
				let pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				pathString = resolve(fileDescriptor.realPath, pathString);
				getFs(this).rmdirSync(pathString);
				return 0;
			}, async function path_remove_directory(fd, path, path_len) {
				path = Number(path);
				path_len = Number(path_len);
				if (path === 0) return 28;
				const { HEAPU8 } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_REMOVE_DIRECTORY, BigInt(0));
				let pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				pathString = resolve(fileDescriptor.realPath, pathString);
				await getFs(this).promises.rmdir(pathString);
				return 0;
			}, [
				"i32",
				"i32",
				"i32"
			], ["i32"]);
			defineImport("path_rename", function path_rename(old_fd, old_path, old_path_len, new_fd, new_path, new_path_len) {
				old_path = Number(old_path);
				old_path_len = Number(old_path_len);
				new_path = Number(new_path);
				new_path_len = Number(new_path_len);
				if (old_path === 0 || new_path === 0) return 28;
				const wasi = _wasi.get(this);
				let oldWrap;
				let newWrap;
				if (old_fd === new_fd) oldWrap = newWrap = wasi.fds.get(old_fd, WasiRights.PATH_RENAME_SOURCE | WasiRights.PATH_RENAME_TARGET, BigInt(0));
				else {
					oldWrap = wasi.fds.get(old_fd, WasiRights.PATH_RENAME_SOURCE, BigInt(0));
					newWrap = wasi.fds.get(new_fd, WasiRights.PATH_RENAME_TARGET, BigInt(0));
				}
				const { HEAPU8 } = getMemory(this);
				const resolvedOldPath = resolve(oldWrap.realPath, decoder.decode(unsharedSlice(HEAPU8, old_path, old_path + old_path_len)));
				const resolvedNewPath = resolve(newWrap.realPath, decoder.decode(unsharedSlice(HEAPU8, new_path, new_path + new_path_len)));
				getFs(this).renameSync(resolvedOldPath, resolvedNewPath);
				return 0;
			}, async function path_rename(old_fd, old_path, old_path_len, new_fd, new_path, new_path_len) {
				old_path = Number(old_path);
				old_path_len = Number(old_path_len);
				new_path = Number(new_path);
				new_path_len = Number(new_path_len);
				if (old_path === 0 || new_path === 0) return 28;
				const wasi = _wasi.get(this);
				let oldWrap;
				let newWrap;
				if (old_fd === new_fd) oldWrap = newWrap = wasi.fds.get(old_fd, WasiRights.PATH_RENAME_SOURCE | WasiRights.PATH_RENAME_TARGET, BigInt(0));
				else {
					oldWrap = wasi.fds.get(old_fd, WasiRights.PATH_RENAME_SOURCE, BigInt(0));
					newWrap = wasi.fds.get(new_fd, WasiRights.PATH_RENAME_TARGET, BigInt(0));
				}
				const { HEAPU8 } = getMemory(this);
				const resolvedOldPath = resolve(oldWrap.realPath, decoder.decode(unsharedSlice(HEAPU8, old_path, old_path + old_path_len)));
				const resolvedNewPath = resolve(newWrap.realPath, decoder.decode(unsharedSlice(HEAPU8, new_path, new_path + new_path_len)));
				await getFs(this).promises.rename(resolvedOldPath, resolvedNewPath);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i32",
				"i32",
				"i32"
			], ["i32"]);
			defineImport("path_symlink", function path_symlink(old_path, old_path_len, fd, new_path, new_path_len) {
				old_path = Number(old_path);
				old_path_len = Number(old_path_len);
				new_path = Number(new_path);
				new_path_len = Number(new_path_len);
				if (old_path === 0 || new_path === 0) return 28;
				const { HEAPU8 } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_SYMLINK, BigInt(0));
				const oldPath = decoder.decode(unsharedSlice(HEAPU8, old_path, old_path + old_path_len));
				if (oldPath.length > 0 && oldPath[0] === "/") return 63;
				let newPath = decoder.decode(unsharedSlice(HEAPU8, new_path, new_path + new_path_len));
				newPath = resolve(fileDescriptor.realPath, newPath);
				getFs(this).symlinkSync(oldPath, newPath);
				return 0;
			}, async function path_symlink(old_path, old_path_len, fd, new_path, new_path_len) {
				old_path = Number(old_path);
				old_path_len = Number(old_path_len);
				new_path = Number(new_path);
				new_path_len = Number(new_path_len);
				if (old_path === 0 || new_path === 0) return 28;
				const { HEAPU8 } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_SYMLINK, BigInt(0));
				const oldPath = decoder.decode(unsharedSlice(HEAPU8, old_path, old_path + old_path_len));
				let newPath = decoder.decode(unsharedSlice(HEAPU8, new_path, new_path + new_path_len));
				newPath = resolve(fileDescriptor.realPath, newPath);
				await getFs(this).promises.symlink(oldPath, newPath);
				return 0;
			}, [
				"i32",
				"i32",
				"i32",
				"i32",
				"i32"
			], ["i32"]);
			defineImport("path_unlink_file", function path_unlink_file(fd, path, path_len) {
				path = Number(path);
				path_len = Number(path_len);
				if (path === 0) return 28;
				const { HEAPU8 } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_UNLINK_FILE, BigInt(0));
				let pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				pathString = resolve(fileDescriptor.realPath, pathString);
				getFs(this).unlinkSync(pathString);
				return 0;
			}, async function path_unlink_file(fd, path, path_len) {
				path = Number(path);
				path_len = Number(path_len);
				if (path === 0) return 28;
				const { HEAPU8 } = getMemory(this);
				const fileDescriptor = _wasi.get(this).fds.get(fd, WasiRights.PATH_UNLINK_FILE, BigInt(0));
				let pathString = decoder.decode(unsharedSlice(HEAPU8, path, path + path_len));
				pathString = resolve(fileDescriptor.realPath, pathString);
				await getFs(this).promises.unlink(pathString);
				return 0;
			}, [
				"i32",
				"i32",
				"i32"
			], ["i32"]);
			this._setMemory = function setMemory(m) {
				if (!(m instanceof _WebAssembly.Memory)) throw new TypeError("\"instance.exports.memory\" property must be a WebAssembly.Memory");
				_memory.set(_this, extendMemory(m));
			};
		}
		static createSync(args, env, preopens, stdio, fs$1, print, printErr) {
			const fds = new SyncTable({
				size: 3,
				in: stdio[0],
				out: stdio[1],
				err: stdio[2],
				fs: fs$1,
				print,
				printErr
			});
			const _this = new WASI$1(args, env, fds, false, fs$1);
			if (preopens.length > 0) for (let i = 0; i < preopens.length; ++i) {
				const realPath = fs$1.realpathSync(preopens[i].realPath, "utf8");
				const fd = fs$1.openSync(realPath, "r", 438);
				fds.insertPreopen(fd, preopens[i].mappedPath, realPath);
			}
			return _this;
		}
		static async createAsync(args, env, preopens, stdio, fs$1, print, printErr, asyncify) {
			const fds = new AsyncTable({
				size: 3,
				in: stdio[0],
				out: stdio[1],
				err: stdio[2],
				print,
				printErr
			});
			const _this = new WASI$1(args, env, fds, true, fs$1, asyncify);
			if (preopens.length > 0) for (let i = 0; i < preopens.length; ++i) {
				const entry = preopens[i];
				const realPath = await fs$1.promises.realpath(entry.realPath);
				const fd = await fs$1.promises.open(realPath, "r", 438);
				await fds.insertPreopen(fd, entry.mappedPath, realPath);
			}
			return _this;
		}
	};
	kEmptyObject = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.create(null));
	kExitCode = Symbol("kExitCode");
	kSetMemory = Symbol("kSetMemory");
	kStarted = Symbol("kStarted");
	kInstance = Symbol("kInstance");
	kBindingName = Symbol("kBindingName");
	WASI$2 = class {
		constructor(options = kEmptyObject) {
			const { args, env, preopens, stdio, _WASI } = validateOptions.call(this, options);
			const wrap$2 = _WASI.createSync(args, env, preopens, stdio, options.fs, options.print, options.printErr);
			const setMemory = wrap$2._setMemory;
			delete wrap$2._setMemory;
			initWASI.call(this, setMemory, wrap$2);
			if (options.returnOnExit) wrap$2.proc_exit = wasiReturnOnProcExit.bind(this);
		}
		finalizeBindings(instance, _a) {
			var _b;
			var { memory = (_b = instance === null || instance === void 0 ? void 0 : instance.exports) === null || _b === void 0 ? void 0 : _b.memory } = _a === void 0 ? {} : _a;
			if (this[kStarted]) throw new Error("WASI instance has already started");
			validateObject(instance, "instance");
			validateObject(instance.exports, "instance.exports");
			this[kSetMemory](memory);
			this[kInstance] = instance;
			this[kStarted] = true;
		}
		start(instance) {
			this.finalizeBindings(instance);
			const { _start, _initialize } = this[kInstance].exports;
			validateFunction(_start, "instance.exports._start");
			validateUndefined(_initialize, "instance.exports._initialize");
			let ret;
			try {
				ret = _start();
			} catch (err) {
				if (err !== kExitCode) throw err;
			}
			if (ret instanceof Promise) return ret.then(() => this[kExitCode], (err) => {
				if (err !== kExitCode) throw err;
				return this[kExitCode];
			});
			return this[kExitCode];
		}
		initialize(instance) {
			this.finalizeBindings(instance);
			const { _start, _initialize } = this[kInstance].exports;
			validateUndefined(_start, "instance.exports._start");
			if (_initialize !== void 0) {
				validateFunction(_initialize, "instance.exports._initialize");
				return _initialize();
			}
		}
		getImportObject() {
			return { [this[kBindingName]]: this.wasiImport };
		}
	};
}) });

//#endregion
//#region ../../node_modules/.pnpm/@napi-rs+wasm-runtime@1.0.5/node_modules/@napi-rs/wasm-runtime/dist/fs-proxy.cjs
var require_fs_proxy = /* @__PURE__ */ __commonJS({ "../../node_modules/.pnpm/@napi-rs+wasm-runtime@1.0.5/node_modules/@napi-rs/wasm-runtime/dist/fs-proxy.cjs": ((exports) => {
	/**
	* @param {unknown} value
	*/
	const getType = (value) => {
		if (value === void 0) return 0;
		if (value === null) return 1;
		const t = typeof value;
		if (t === "boolean") return 2;
		if (t === "number") return 3;
		if (t === "string") return 4;
		if (t === "object") return 6;
		if (t === "bigint") return 9;
		return -1;
	};
	/**
	* @param {import('memfs').IFs} memfs
	* @param {any} value
	* @param {ReturnType<typeof getType>} type
	* @returns {Uint8Array}
	*/
	const encodeValue = (memfs, value, type) => {
		switch (type) {
			case 0:
			case 1: return new Uint8Array(0);
			case 2: {
				const view = new Int32Array(1);
				view[0] = value ? 1 : 0;
				return new Uint8Array(view.buffer);
			}
			case 3: {
				const view = new Float64Array(1);
				view[0] = value;
				return new Uint8Array(view.buffer);
			}
			case 4: return new TextEncoder().encode(value);
			case 6: {
				function storeConstructor(obj, memfs$1, processed = /* @__PURE__ */ new WeakSet()) {
					if (!obj || typeof obj !== "object") return;
					if (processed.has(obj)) return;
					processed.add(obj);
					const [entry] = Object.entries(memfs$1).filter(([_, v]) => v === obj.constructor)[0] ?? [];
					if (entry) Object.defineProperty(obj, "__constructor__", {
						configurable: true,
						writable: true,
						enumerable: true,
						value: entry
					});
					for (const value$1 of Object.values(obj)) storeConstructor(value$1, memfs$1, processed);
				}
				storeConstructor(value, memfs);
				const json = JSON.stringify(value, (_, value$1) => {
					if (typeof value$1 === "bigint") return `BigInt(${String(value$1)})`;
					if (value$1 instanceof Error) return {
						...value$1,
						message: value$1.message,
						stack: value$1.stack,
						__error__: value$1.constructor.name
					};
					return value$1;
				});
				return new TextEncoder().encode(json);
			}
			case 9: {
				const view = new BigInt64Array(1);
				view[0] = value;
				return new Uint8Array(view.buffer);
			}
			case -1:
			default: throw new Error("unsupported data");
		}
	};
	/**
	* @param {typeof import('memfs')} memfs
	* @param {Uint8Array} payload
	* @param {number} type
	* @returns {any}
	*/
	const decodeValue = (memfs, payload, type) => {
		if (type === 0) return void 0;
		if (type === 1) return null;
		if (type === 2) return Boolean(new Int32Array(payload.buffer, payload.byteOffset, 1)[0]);
		if (type === 3) return new Float64Array(payload.buffer, payload.byteOffset, 1)[0];
		if (type === 4) return new TextDecoder().decode(payload.slice());
		if (type === 6) {
			const obj = JSON.parse(new TextDecoder().decode(payload.slice()), (_key, value) => {
				if (typeof value === "string") {
					const matched = value.match(/^BigInt\((-?\d+)\)$/);
					if (matched && matched[1]) return BigInt(matched[1]);
				}
				return value;
			});
			function loadConstructor(obj$1, memfs$1, processed = /* @__PURE__ */ new WeakSet()) {
				if (!obj$1 || typeof obj$1 !== "object") return;
				if (processed.has(obj$1)) return;
				processed.add(obj$1);
				if (obj$1.__constructor__) {
					const ctor = obj$1.__constructor__;
					delete obj$1.__constructor__;
					Object.setPrototypeOf(obj$1, memfs$1[ctor].prototype);
				}
				for (const value of Object.values(obj$1)) loadConstructor(value, memfs$1, processed);
			}
			loadConstructor(obj, memfs);
			if (obj.__error__) {
				const name = obj.__error__;
				const ErrorConstructor = globalThis[name] || Error;
				delete obj.__error__;
				const err = new ErrorConstructor(obj.message);
				Object.defineProperty(err, "stack", {
					configurable: true,
					enumerable: false,
					writable: true,
					value: err.stack
				});
				Object.defineProperty(err, Symbol.toStringTag, {
					configurable: true,
					enumerable: false,
					writable: true,
					value: name
				});
				for (const [k, v] of Object.entries(obj)) {
					if (k === "message" || k === "stack") continue;
					err[k] = v;
				}
				return err;
			}
			return obj;
		}
		if (type === 9) return new BigInt64Array(payload.buffer, payload.byteOffset, 1)[0];
		throw new Error("unsupported data");
	};
	/**
	* @param {import('memfs').IFs} fs
	* @returns {(e: { data: { __fs__: { sab: Int32Array, type: keyof import('memfs').IFs, payload: any[] } } }) => void}
	*/
	const createOnMessage$1 = (fs$1) => function onMessage(e) {
		if (e.data.__fs__) {
			/**
			* 0..4                    status(int32_t):        21(waiting) 0(success) 1(error)
			* 5..8                    type(napi_valuetype):   0(undefined) 1(null) 2(boolean) 3(number) 4(string) 6(jsonstring) 9(bigint) -1(unsupported)
			* 9..16                   payload_size(uint32_t)  <= 1024
			* 16..16 + payload_size   payload_content
			*/
			const { sab, type, payload } = e.data.__fs__;
			const fn = fs$1[type];
			try {
				const ret = fn.apply(fs$1, payload);
				const t = getType(ret);
				Atomics.store(sab, 1, t);
				const v = encodeValue(fs$1, ret, t);
				Atomics.store(sab, 2, v.length);
				new Uint8Array(sab.buffer).set(v, 16);
				Atomics.store(sab, 0, 0);
			} catch (err) {
				const t = getType(err);
				Atomics.store(sab, 1, t);
				const v = encodeValue(fs$1, err, t);
				Atomics.store(sab, 2, v.length);
				new Uint8Array(sab.buffer).set(v, 16);
				Atomics.store(sab, 0, 1);
			} finally {
				Atomics.notify(sab, 0);
			}
		}
	};
	/**
	* @param {typeof import('memfs')} memfs
	*/
	const createFsProxy$1 = (memfs) => new Proxy({}, { get(_target, p, _receiver) {
		/**
		* @param {any[]} args
		*/
		return function(...args) {
			const sab = new SharedArrayBuffer(10256);
			const i32arr = new Int32Array(sab);
			Atomics.store(i32arr, 0, 21);
			postMessage({ __fs__: {
				sab: i32arr,
				type: p,
				payload: args
			} });
			Atomics.wait(i32arr, 0, 21);
			const status = Atomics.load(i32arr, 0);
			const type = Atomics.load(i32arr, 1);
			const size = Atomics.load(i32arr, 2);
			const content = new Uint8Array(sab, 16, size);
			const value = decodeValue(memfs, content, type);
			if (status === 1) throw value;
			return value;
		};
	} });
	exports.createFsProxy = createFsProxy$1;
	exports.createOnMessage = createOnMessage$1;
}) });

//#endregion
//#region ../../node_modules/.pnpm/@napi-rs+wasm-runtime@1.0.5/node_modules/@napi-rs/wasm-runtime/runtime.cjs
var require_runtime = /* @__PURE__ */ __commonJS({ "../../node_modules/.pnpm/@napi-rs+wasm-runtime@1.0.5/node_modules/@napi-rs/wasm-runtime/runtime.cjs": ((exports, module) => {
	const { MessageHandler, instantiateNapiModuleSync, instantiateNapiModule } = (init_emnapi_core_esm_bundler(), __toCommonJS(emnapi_core_esm_bundler_exports));
	const { getDefaultContext } = (init_emnapi_esm_bundler(), __toCommonJS(emnapi_esm_bundler_exports));
	const { WASI } = (init_wasm_util_esm_bundler(), __toCommonJS(wasm_util_esm_bundler_exports));
	const { createFsProxy, createOnMessage } = require_fs_proxy();
	module.exports = {
		MessageHandler,
		instantiateNapiModule,
		instantiateNapiModuleSync,
		getDefaultContext,
		WASI,
		createFsProxy,
		createOnMessage
	};
}) });

//#endregion
//#region src/rolldown-binding.wasi.cjs
var require_rolldown_binding_wasi = /* @__PURE__ */ __commonJS({ "src/rolldown-binding.wasi.cjs": ((exports, module) => {
	const __nodeFs = __require("node:fs");
	const __nodePath = __require("node:path");
	const { WASI: __nodeWASI } = __require("node:wasi");
	const { Worker } = __require("node:worker_threads");
	const { createOnMessage: __wasmCreateOnMessageForFsProxy, getDefaultContext: __emnapiGetDefaultContext, instantiateNapiModuleSync: __emnapiInstantiateNapiModuleSync } = require_runtime();
	const __rootDir = __nodePath.parse(process.cwd()).root;
	const __wasi = new __nodeWASI({
		version: "preview1",
		env: process.env,
		preopens: { [__rootDir]: __rootDir }
	});
	const __emnapiContext = __emnapiGetDefaultContext();
	const __sharedMemory = new WebAssembly.Memory({
		initial: 16384,
		maximum: 65536,
		shared: true
	});
	let __wasmFilePath = __nodePath.join(__dirname, "rolldown-binding.wasm32-wasi.wasm");
	const __wasmDebugFilePath = __nodePath.join(__dirname, "rolldown-binding.wasm32-wasi.debug.wasm");
	if (__nodeFs.existsSync(__wasmDebugFilePath)) __wasmFilePath = __wasmDebugFilePath;
	else if (!__nodeFs.existsSync(__wasmFilePath)) try {
		__wasmFilePath = __nodePath.resolve("@rolldown/binding-wasm32-wasi");
	} catch {
		throw new Error("Cannot find rolldown-binding.wasm32-wasi.wasm file, and @rolldown/binding-wasm32-wasi package is not installed.");
	}
	const { instance: __napiInstance, module: __wasiModule, napiModule: __napiModule } = __emnapiInstantiateNapiModuleSync(__nodeFs.readFileSync(__wasmFilePath), {
		context: __emnapiContext,
		asyncWorkPoolSize: (function() {
			const threadsSizeFromEnv = Number(process.env.NAPI_RS_ASYNC_WORK_POOL_SIZE ?? process.env.UV_THREADPOOL_SIZE);
			if (threadsSizeFromEnv > 0) return threadsSizeFromEnv;
			else return 4;
		})(),
		reuseWorker: true,
		wasi: __wasi,
		onCreateWorker() {
			const worker = new Worker(__nodePath.join(__dirname, "wasi-worker.mjs"), { env: process.env });
			worker.onmessage = ({ data }) => {
				__wasmCreateOnMessageForFsProxy(__nodeFs)(data);
			};
			{
				const kPublicPort = Object.getOwnPropertySymbols(worker).find((s) => s.toString().includes("kPublicPort"));
				if (kPublicPort) worker[kPublicPort].ref = () => {};
				const kHandle = Object.getOwnPropertySymbols(worker).find((s) => s.toString().includes("kHandle"));
				if (kHandle) worker[kHandle].ref = () => {};
				worker.unref();
			}
			return worker;
		},
		overwriteImports(importObject) {
			importObject.env = {
				...importObject.env,
				...importObject.napi,
				...importObject.emnapi,
				memory: __sharedMemory
			};
			return importObject;
		},
		beforeInit({ instance }) {
			for (const name of Object.keys(instance.exports)) if (name.startsWith("__napi_register__")) instance.exports[name]();
		}
	});
	module.exports = __napiModule.exports;
	module.exports.minify = __napiModule.exports.minify;
	module.exports.Severity = __napiModule.exports.Severity;
	module.exports.ParseResult = __napiModule.exports.ParseResult;
	module.exports.ExportExportNameKind = __napiModule.exports.ExportExportNameKind;
	module.exports.ExportImportNameKind = __napiModule.exports.ExportImportNameKind;
	module.exports.ExportLocalNameKind = __napiModule.exports.ExportLocalNameKind;
	module.exports.ImportNameKind = __napiModule.exports.ImportNameKind;
	module.exports.parseAsync = __napiModule.exports.parseAsync;
	module.exports.parseSync = __napiModule.exports.parseSync;
	module.exports.rawTransferSupported = __napiModule.exports.rawTransferSupported;
	module.exports.ResolverFactory = __napiModule.exports.ResolverFactory;
	module.exports.EnforceExtension = __napiModule.exports.EnforceExtension;
	module.exports.ModuleType = __napiModule.exports.ModuleType;
	module.exports.sync = __napiModule.exports.sync;
	module.exports.HelperMode = __napiModule.exports.HelperMode;
	module.exports.isolatedDeclaration = __napiModule.exports.isolatedDeclaration;
	module.exports.moduleRunnerTransform = __napiModule.exports.moduleRunnerTransform;
	module.exports.transform = __napiModule.exports.transform;
	module.exports.transformAsync = __napiModule.exports.transformAsync;
	module.exports.BindingBundleEndEventData = __napiModule.exports.BindingBundleEndEventData;
	module.exports.BindingBundleErrorEventData = __napiModule.exports.BindingBundleErrorEventData;
	module.exports.BindingBundler = __napiModule.exports.BindingBundler;
	module.exports.BindingBundlerImpl = __napiModule.exports.BindingBundlerImpl;
	module.exports.BindingCallableBuiltinPlugin = __napiModule.exports.BindingCallableBuiltinPlugin;
	module.exports.BindingChunkingContext = __napiModule.exports.BindingChunkingContext;
	module.exports.BindingClientHmrUpdate = __napiModule.exports.BindingClientHmrUpdate;
	module.exports.BindingDevEngine = __napiModule.exports.BindingDevEngine;
	module.exports.BindingHmrOutput = __napiModule.exports.BindingHmrOutput;
	module.exports.BindingMagicString = __napiModule.exports.BindingMagicString;
	module.exports.BindingModuleInfo = __napiModule.exports.BindingModuleInfo;
	module.exports.BindingNormalizedOptions = __napiModule.exports.BindingNormalizedOptions;
	module.exports.BindingOutputAsset = __napiModule.exports.BindingOutputAsset;
	module.exports.BindingOutputChunk = __napiModule.exports.BindingOutputChunk;
	module.exports.BindingOutputs = __napiModule.exports.BindingOutputs;
	module.exports.BindingPluginContext = __napiModule.exports.BindingPluginContext;
	module.exports.BindingRenderedChunk = __napiModule.exports.BindingRenderedChunk;
	module.exports.BindingRenderedChunkMeta = __napiModule.exports.BindingRenderedChunkMeta;
	module.exports.BindingRenderedModule = __napiModule.exports.BindingRenderedModule;
	module.exports.BindingTransformPluginContext = __napiModule.exports.BindingTransformPluginContext;
	module.exports.BindingWatcher = __napiModule.exports.BindingWatcher;
	module.exports.BindingWatcherChangeData = __napiModule.exports.BindingWatcherChangeData;
	module.exports.BindingWatcherEvent = __napiModule.exports.BindingWatcherEvent;
	module.exports.ParallelJsPluginRegistry = __napiModule.exports.ParallelJsPluginRegistry;
	module.exports.ScheduledBuild = __napiModule.exports.ScheduledBuild;
	module.exports.TraceSubscriberGuard = __napiModule.exports.TraceSubscriberGuard;
	module.exports.BindingAttachDebugInfo = __napiModule.exports.BindingAttachDebugInfo;
	module.exports.BindingBuiltinPluginName = __napiModule.exports.BindingBuiltinPluginName;
	module.exports.BindingChunkModuleOrderBy = __napiModule.exports.BindingChunkModuleOrderBy;
	module.exports.BindingJsx = __napiModule.exports.BindingJsx;
	module.exports.BindingLogLevel = __napiModule.exports.BindingLogLevel;
	module.exports.BindingPluginOrder = __napiModule.exports.BindingPluginOrder;
	module.exports.BindingPropertyReadSideEffects = __napiModule.exports.BindingPropertyReadSideEffects;
	module.exports.BindingPropertyWriteSideEffects = __napiModule.exports.BindingPropertyWriteSideEffects;
	module.exports.FilterTokenKind = __napiModule.exports.FilterTokenKind;
	module.exports.initTraceSubscriber = __napiModule.exports.initTraceSubscriber;
	module.exports.registerPlugins = __napiModule.exports.registerPlugins;
	module.exports.shutdownAsyncRuntime = __napiModule.exports.shutdownAsyncRuntime;
	module.exports.startAsyncRuntime = __napiModule.exports.startAsyncRuntime;
	module.exports.JsWatcher = __napiModule.exports.JsWatcher;
}) });

//#endregion
//#region src/webcontainer-fallback.cjs
var require_webcontainer_fallback = /* @__PURE__ */ __commonJS({ "src/webcontainer-fallback.cjs": ((exports, module) => {
	const fs = __require("node:fs");
	const childProcess = __require("node:child_process");
	const version = JSON.parse(fs.readFileSync(__require.resolve("rolldown/package.json"), "utf-8")).version;
	const baseDir = `/tmp/rolldown-${version}`;
	const bindingEntry = `${baseDir}/node_modules/@rolldown/binding-wasm32-wasi/rolldown-binding.wasi.cjs`;
	if (!fs.existsSync(bindingEntry)) {
		const bindingPkg = `@rolldown/binding-wasm32-wasi@${version}`;
		fs.rmSync(baseDir, {
			recursive: true,
			force: true
		});
		fs.mkdirSync(baseDir, { recursive: true });
		console.log(`[rolldown] Downloading ${bindingPkg} on WebContainer...`);
		childProcess.execFileSync("pnpm", ["i", bindingPkg], {
			cwd: baseDir,
			stdio: "inherit"
		});
	}
	module.exports = __require(bindingEntry);
}) });

//#endregion
//#region src/binding.js
new URL(".", import.meta.url).pathname;
const { readFileSync } = __require("node:fs");
let nativeBinding = null;
const loadErrors = [];
const isMusl = () => {
	let musl = false;
	if (process.platform === "linux") {
		musl = isMuslFromFilesystem();
		if (musl === null) musl = isMuslFromReport();
		if (musl === null) musl = isMuslFromChildProcess();
	}
	return musl;
};
const isFileMusl = (f) => f.includes("libc.musl-") || f.includes("ld-musl-");
const isMuslFromFilesystem = () => {
	try {
		return readFileSync("/usr/bin/ldd", "utf-8").includes("musl");
	} catch {
		return null;
	}
};
const isMuslFromReport = () => {
	let report = null;
	if (typeof process.report?.getReport === "function") {
		process.report.excludeNetwork = true;
		report = process.report.getReport();
	}
	if (!report) return null;
	if (report.header && report.header.glibcVersionRuntime) return false;
	if (Array.isArray(report.sharedObjects)) {
		if (report.sharedObjects.some(isFileMusl)) return true;
	}
	return false;
};
const isMuslFromChildProcess = () => {
	try {
		return __require("child_process").execSync("ldd --version", { encoding: "utf8" }).includes("musl");
	} catch (e) {
		return false;
	}
};
function requireNative() {
	if (process.env.NAPI_RS_NATIVE_LIBRARY_PATH) try {
		return __require(process.env.NAPI_RS_NATIVE_LIBRARY_PATH);
	} catch (err) {
		loadErrors.push(err);
	}
	else if (process.platform === "android") if (process.arch === "arm64") {
		try {
			return __require("../rolldown-binding.android-arm64.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-android-arm64");
			const bindingPackageVersion = __require("@rolldown/binding-android-arm64/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else if (process.arch === "arm") {
		try {
			return __require("../rolldown-binding.android-arm-eabi.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-android-arm-eabi");
			const bindingPackageVersion = __require("@rolldown/binding-android-arm-eabi/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else loadErrors.push(/* @__PURE__ */ new Error(`Unsupported architecture on Android ${process.arch}`));
	else if (process.platform === "win32") if (process.arch === "x64") {
		try {
			return __require("../rolldown-binding.win32-x64-msvc.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-win32-x64-msvc");
			const bindingPackageVersion = __require("@rolldown/binding-win32-x64-msvc/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else if (process.arch === "ia32") {
		try {
			return __require("../rolldown-binding.win32-ia32-msvc.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-win32-ia32-msvc");
			const bindingPackageVersion = __require("@rolldown/binding-win32-ia32-msvc/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else if (process.arch === "arm64") {
		try {
			return __require("../rolldown-binding.win32-arm64-msvc.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-win32-arm64-msvc");
			const bindingPackageVersion = __require("@rolldown/binding-win32-arm64-msvc/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else loadErrors.push(/* @__PURE__ */ new Error(`Unsupported architecture on Windows: ${process.arch}`));
	else if (process.platform === "darwin") {
		try {
			return __require("../rolldown-binding.darwin-universal.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-darwin-universal");
			const bindingPackageVersion = __require("@rolldown/binding-darwin-universal/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
		if (process.arch === "x64") {
			try {
				return __require("../rolldown-binding.darwin-x64.node");
			} catch (e) {
				loadErrors.push(e);
			}
			try {
				const binding = __require("@rolldown/binding-darwin-x64");
				const bindingPackageVersion = __require("@rolldown/binding-darwin-x64/package.json").version;
				if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
				return binding;
			} catch (e) {
				loadErrors.push(e);
			}
		} else if (process.arch === "arm64") {
			try {
				return __require("../rolldown-binding.darwin-arm64.node");
			} catch (e) {
				loadErrors.push(e);
			}
			try {
				const binding = __require("@rolldown/binding-darwin-arm64");
				const bindingPackageVersion = __require("@rolldown/binding-darwin-arm64/package.json").version;
				if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
				return binding;
			} catch (e) {
				loadErrors.push(e);
			}
		} else loadErrors.push(/* @__PURE__ */ new Error(`Unsupported architecture on macOS: ${process.arch}`));
	} else if (process.platform === "freebsd") if (process.arch === "x64") {
		try {
			return __require("../rolldown-binding.freebsd-x64.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-freebsd-x64");
			const bindingPackageVersion = __require("@rolldown/binding-freebsd-x64/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else if (process.arch === "arm64") {
		try {
			return __require("../rolldown-binding.freebsd-arm64.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-freebsd-arm64");
			const bindingPackageVersion = __require("@rolldown/binding-freebsd-arm64/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else loadErrors.push(/* @__PURE__ */ new Error(`Unsupported architecture on FreeBSD: ${process.arch}`));
	else if (process.platform === "linux") if (process.arch === "x64") if (isMusl()) {
		try {
			return __require("../rolldown-binding.linux-x64-musl.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-x64-musl");
			const bindingPackageVersion = __require("@rolldown/binding-linux-x64-musl/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else {
		try {
			return __require("../rolldown-binding.linux-x64-gnu.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-x64-gnu");
			const bindingPackageVersion = __require("@rolldown/binding-linux-x64-gnu/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	}
	else if (process.arch === "arm64") if (isMusl()) {
		try {
			return __require("../rolldown-binding.linux-arm64-musl.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-arm64-musl");
			const bindingPackageVersion = __require("@rolldown/binding-linux-arm64-musl/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else {
		try {
			return __require("../rolldown-binding.linux-arm64-gnu.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-arm64-gnu");
			const bindingPackageVersion = __require("@rolldown/binding-linux-arm64-gnu/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	}
	else if (process.arch === "arm") if (isMusl()) {
		try {
			return __require("../rolldown-binding.linux-arm-musleabihf.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-arm-musleabihf");
			const bindingPackageVersion = __require("@rolldown/binding-linux-arm-musleabihf/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else {
		try {
			return __require("../rolldown-binding.linux-arm-gnueabihf.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-arm-gnueabihf");
			const bindingPackageVersion = __require("@rolldown/binding-linux-arm-gnueabihf/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	}
	else if (process.arch === "loong64") if (isMusl()) {
		try {
			return __require("../rolldown-binding.linux-loong64-musl.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-loong64-musl");
			const bindingPackageVersion = __require("@rolldown/binding-linux-loong64-musl/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else {
		try {
			return __require("../rolldown-binding.linux-loong64-gnu.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-loong64-gnu");
			const bindingPackageVersion = __require("@rolldown/binding-linux-loong64-gnu/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	}
	else if (process.arch === "riscv64") if (isMusl()) {
		try {
			return __require("../rolldown-binding.linux-riscv64-musl.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-riscv64-musl");
			const bindingPackageVersion = __require("@rolldown/binding-linux-riscv64-musl/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else {
		try {
			return __require("../rolldown-binding.linux-riscv64-gnu.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-riscv64-gnu");
			const bindingPackageVersion = __require("@rolldown/binding-linux-riscv64-gnu/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	}
	else if (process.arch === "ppc64") {
		try {
			return __require("../rolldown-binding.linux-ppc64-gnu.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-ppc64-gnu");
			const bindingPackageVersion = __require("@rolldown/binding-linux-ppc64-gnu/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else if (process.arch === "s390x") {
		try {
			return __require("../rolldown-binding.linux-s390x-gnu.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-linux-s390x-gnu");
			const bindingPackageVersion = __require("@rolldown/binding-linux-s390x-gnu/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else loadErrors.push(/* @__PURE__ */ new Error(`Unsupported architecture on Linux: ${process.arch}`));
	else if (process.platform === "openharmony") if (process.arch === "arm64") {
		try {
			return __require("../rolldown-binding.openharmony-arm64.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-openharmony-arm64");
			const bindingPackageVersion = __require("@rolldown/binding-openharmony-arm64/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else if (process.arch === "x64") {
		try {
			return __require("../rolldown-binding.openharmony-x64.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-openharmony-x64");
			const bindingPackageVersion = __require("@rolldown/binding-openharmony-x64/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else if (process.arch === "arm") {
		try {
			return __require("../rolldown-binding.openharmony-arm.node");
		} catch (e) {
			loadErrors.push(e);
		}
		try {
			const binding = __require("@rolldown/binding-openharmony-arm");
			const bindingPackageVersion = __require("@rolldown/binding-openharmony-arm/package.json").version;
			if (bindingPackageVersion !== "1.0.0-beta.41" && process.env.NAPI_RS_ENFORCE_VERSION_CHECK && process.env.NAPI_RS_ENFORCE_VERSION_CHECK !== "0") throw new Error(`Native binding package version mismatch, expected 1.0.0-beta.41 but got ${bindingPackageVersion}. You can reinstall dependencies to fix this issue.`);
			return binding;
		} catch (e) {
			loadErrors.push(e);
		}
	} else loadErrors.push(/* @__PURE__ */ new Error(`Unsupported architecture on OpenHarmony: ${process.arch}`));
	else loadErrors.push(/* @__PURE__ */ new Error(`Unsupported OS: ${process.platform}, architecture: ${process.arch}`));
}
nativeBinding = requireNative();
if (!nativeBinding || process.env.NAPI_RS_FORCE_WASI) {
	let wasiBinding = null;
	let wasiBindingError = null;
	try {
		wasiBinding = require_rolldown_binding_wasi();
		nativeBinding = wasiBinding;
	} catch (err) {
		if (process.env.NAPI_RS_FORCE_WASI) wasiBindingError = err;
	}
	if (!nativeBinding) try {
		wasiBinding = __require("@rolldown/binding-wasm32-wasi");
		nativeBinding = wasiBinding;
	} catch (err) {
		if (process.env.NAPI_RS_FORCE_WASI) {
			wasiBindingError.cause = err;
			loadErrors.push(err);
		}
	}
	if (process.env.NAPI_RS_FORCE_WASI === "error" && !wasiBinding) {
		const error$1 = /* @__PURE__ */ new Error("WASI binding not found and NAPI_RS_FORCE_WASI is set to error");
		error$1.cause = wasiBindingError;
		throw error$1;
	}
}
if (!nativeBinding && globalThis.process?.versions?.["webcontainer"]) try {
	nativeBinding = require_webcontainer_fallback();
} catch (err) {
	loadErrors.push(err);
}
if (!nativeBinding) {
	if (loadErrors.length > 0) throw new Error("Cannot find native binding. npm has a bug related to optional dependencies (https://github.com/npm/cli/issues/4828). Please try `npm i` again after removing both package-lock.json and node_modules directory.", { cause: loadErrors.reduce((err, cur) => {
		cur.cause = err;
		return cur;
	}) });
	throw new Error(`Failed to load native binding`);
}
const { minify, Severity, ParseResult, ExportExportNameKind, ExportImportNameKind, ExportLocalNameKind, ImportNameKind, parseAsync, parseSync, rawTransferSupported, ResolverFactory, EnforceExtension, ModuleType, sync, HelperMode, isolatedDeclaration, moduleRunnerTransform, transform, transformAsync, BindingBundleEndEventData, BindingBundleErrorEventData, BindingBundler, BindingBundlerImpl, BindingCallableBuiltinPlugin, BindingChunkingContext, BindingClientHmrUpdate, BindingDevEngine, BindingHmrOutput, BindingMagicString, BindingModuleInfo, BindingNormalizedOptions, BindingOutputAsset, BindingOutputChunk, BindingOutputs, BindingPluginContext, BindingRenderedChunk, BindingRenderedChunkMeta, BindingRenderedModule, BindingTransformPluginContext, BindingWatcher, BindingWatcherChangeData, BindingWatcherEvent, ParallelJsPluginRegistry, ScheduledBuild, TraceSubscriberGuard, BindingAttachDebugInfo, BindingBuiltinPluginName, BindingChunkModuleOrderBy, BindingJsx, BindingLogLevel, BindingPluginOrder, BindingPropertyReadSideEffects, BindingPropertyWriteSideEffects, FilterTokenKind, initTraceSubscriber, registerPlugins, shutdownAsyncRuntime, startAsyncRuntime, JsWatcher } = nativeBinding;

//#endregion
//#region src/utils/code-frame.ts
function spaces(index) {
	let result = "";
	while (index--) result += " ";
	return result;
}
function tabsToSpaces(value) {
	return value.replace(/^\t+/, (match) => match.split("	").join("  "));
}
const LINE_TRUNCATE_LENGTH = 120;
const MIN_CHARACTERS_SHOWN_AFTER_LOCATION = 10;
const ELLIPSIS = "...";
function getCodeFrame(source, line, column) {
	let lines = source.split("\n");
	if (line > lines.length) return "";
	const maxLineLength = Math.max(tabsToSpaces(lines[line - 1].slice(0, column)).length + MIN_CHARACTERS_SHOWN_AFTER_LOCATION + 3, LINE_TRUNCATE_LENGTH);
	const frameStart = Math.max(0, line - 3);
	let frameEnd = Math.min(line + 2, lines.length);
	lines = lines.slice(frameStart, frameEnd);
	while (!/\S/.test(lines[lines.length - 1])) {
		lines.pop();
		frameEnd -= 1;
	}
	const digits = String(frameEnd).length;
	return lines.map((sourceLine, index) => {
		const isErrorLine = frameStart + index + 1 === line;
		let lineNumber = String(index + frameStart + 1);
		while (lineNumber.length < digits) lineNumber = ` ${lineNumber}`;
		let displayedLine = tabsToSpaces(sourceLine);
		if (displayedLine.length > maxLineLength) displayedLine = `${displayedLine.slice(0, maxLineLength - 3)}${ELLIPSIS}`;
		if (isErrorLine) {
			const indicator = spaces(digits + 2 + tabsToSpaces(sourceLine.slice(0, column)).length) + "^";
			return `${lineNumber}: ${displayedLine}\n${indicator}`;
		}
		return `${lineNumber}: ${displayedLine}`;
	}).join("\n");
}

//#endregion
//#region src/log/locate-character/index.js
/** @typedef {import('./types').Location} Location */
/**
* @param {import('./types').Range} range
* @param {number} index
*/
function rangeContains(range, index) {
	return range.start <= index && index < range.end;
}
/**
* @param {string} source
* @param {import('./types').Options} [options]
*/
function getLocator(source, options = {}) {
	const { offsetLine = 0, offsetColumn = 0 } = options;
	let start = 0;
	const ranges = source.split("\n").map((line, i$1) => {
		const end = start + line.length + 1;
		/** @type {import('./types').Range} */
		const range = {
			start,
			end,
			line: i$1
		};
		start = end;
		return range;
	});
	let i = 0;
	/**
	* @param {string | number} search
	* @param {number} [index]
	* @returns {Location | undefined}
	*/
	function locator(search, index) {
		if (typeof search === "string") search = source.indexOf(search, index ?? 0);
		if (search === -1) return void 0;
		let range = ranges[i];
		const d = search >= range.end ? 1 : -1;
		while (range) {
			if (rangeContains(range, search)) return {
				line: offsetLine + range.line,
				column: offsetColumn + search - range.start,
				character: search
			};
			i += d;
			range = ranges[i];
		}
	}
	return locator;
}
/**
* @param {string} source
* @param {string | number} search
* @param {import('./types').Options} [options]
* @returns {Location | undefined}
*/
function locate(source, search, options) {
	return getLocator(source, options)(search, options && options.startIndex);
}

//#endregion
//#region src/log/logs.ts
const INVALID_LOG_POSITION = "INVALID_LOG_POSITION", PLUGIN_ERROR = "PLUGIN_ERROR", INPUT_HOOK_IN_OUTPUT_PLUGIN = "INPUT_HOOK_IN_OUTPUT_PLUGIN", CYCLE_LOADING = "CYCLE_LOADING", MULTIPLY_NOTIFY_OPTION = "MULTIPLY_NOTIFY_OPTION", PARSE_ERROR = "PARSE_ERROR", DUPLICATE_JSX_CONFIG = "DUPLICATE_JSX_CONFIG", NO_FS_IN_BROWSER = "NO_FS_IN_BROWSER";
function logParseError(message) {
	return {
		code: PARSE_ERROR,
		message
	};
}
function logInvalidLogPosition(pluginName) {
	return {
		code: INVALID_LOG_POSITION,
		message: `Plugin "${pluginName}" tried to add a file position to a log or warning. This is only supported in the "transform" hook at the moment and will be ignored.`
	};
}
function logInputHookInOutputPlugin(pluginName, hookName) {
	return {
		code: INPUT_HOOK_IN_OUTPUT_PLUGIN,
		message: `The "${hookName}" hook used by the output plugin ${pluginName} is a build time hook and will not be run for that plugin. Either this plugin cannot be used as an output plugin, or it should have an option to configure it as an output plugin.`
	};
}
function logCycleLoading(pluginName, moduleId) {
	return {
		code: CYCLE_LOADING,
		message: `Found the module "${moduleId}" cycle loading at ${pluginName} plugin, it maybe blocking fetching modules.`
	};
}
function logMultiplyNotifyOption() {
	return {
		code: MULTIPLY_NOTIFY_OPTION,
		message: `Found multiply notify option at watch options, using first one to start notify watcher.`
	};
}
function logDuplicateJsxConfig() {
	return {
		code: DUPLICATE_JSX_CONFIG,
		message: "Both `options.jsx` and `options.transform.jsx` are set so `options.jsx` is ignored"
	};
}
function logPluginError(error$1, plugin, { hook, id } = {}) {
	try {
		const code = error$1.code;
		if (!error$1.pluginCode && code != null && (typeof code !== "string" || !code.startsWith("PLUGIN_"))) error$1.pluginCode = code;
		error$1.code = PLUGIN_ERROR;
		error$1.plugin = plugin;
		if (hook) error$1.hook = hook;
		if (id) error$1.id = id;
	} catch (_) {} finally {
		return error$1;
	}
}
function error(base) {
	if (!(base instanceof Error)) {
		base = Object.assign(new Error(base.message), base);
		Object.defineProperty(base, "name", {
			value: "RollupError",
			writable: true
		});
	}
	throw base;
}
function augmentCodeLocation(properties, pos, source, id) {
	if (typeof pos === "object") {
		const { line, column } = pos;
		properties.loc = {
			column,
			file: id,
			line
		};
	} else {
		properties.pos = pos;
		const location = locate(source, pos, { offsetLine: 1 });
		if (!location) return;
		const { line, column } = location;
		properties.loc = {
			column,
			file: id,
			line
		};
	}
	if (properties.frame === void 0) {
		const { line, column } = properties.loc;
		properties.frame = getCodeFrame(source, line, column);
	}
}

//#endregion
//#region ../../node_modules/.pnpm/oxc-parser@0.93.0/node_modules/oxc-parser/src-js/wrap.js
function wrap$1(result) {
	let program, module$1, comments, errors;
	return {
		get program() {
			if (!program) program = jsonParseAst(result.program);
			return program;
		},
		get module() {
			if (!module$1) module$1 = result.module;
			return module$1;
		},
		get comments() {
			if (!comments) comments = result.comments;
			return comments;
		},
		get errors() {
			if (!errors) errors = result.errors;
			return errors;
		}
	};
}
function jsonParseAst(programJson) {
	const { node: program, fixes } = JSON.parse(programJson);
	for (const fixPath of fixes) applyFix(program, fixPath);
	return program;
}
function applyFix(program, fixPath) {
	let node = program;
	for (const key of fixPath) node = node[key];
	if (node.bigint) node.value = BigInt(node.bigint);
	else try {
		node.value = RegExp(node.regex.pattern, node.regex.flags);
	} catch (_err) {}
}

//#endregion
//#region src/parse-ast-index.ts
function wrap(result, sourceText) {
	result = wrap$1(result);
	if (result.errors.length > 0) return normalizeParseError(sourceText, result.errors);
	return result.program;
}
function normalizeParseError(sourceText, errors) {
	let message = `Parse failed with ${errors.length} error${errors.length < 2 ? "" : "s"}:\n`;
	for (let i = 0; i < errors.length; i++) {
		if (i >= 5) {
			message += "\n...";
			break;
		}
		const e = errors[i];
		message += e.message + "\n" + e.labels.map((label) => {
			const location = locate(sourceText, label.start, { offsetLine: 1 });
			if (!location) return;
			return getCodeFrame(sourceText, location.line, location.column);
		}).filter(Boolean).join("\n");
	}
	return error(logParseError(message));
}
const defaultParserOptions = {
	lang: "js",
	preserveParens: false
};
function parseAst(sourceText, options, filename) {
	return wrap(parseSync(filename ?? "file.js", sourceText, {
		...defaultParserOptions,
		...options
	}), sourceText);
}
async function parseAstAsync(sourceText, options, filename) {
	return wrap(await parseAsync(filename ?? "file.js", sourceText, {
		...defaultParserOptions,
		...options
	}), sourceText);
}

//#endregion
export { BindingAttachDebugInfo, BindingBundler, BindingCallableBuiltinPlugin, BindingChunkModuleOrderBy, BindingClientHmrUpdate, BindingDevEngine, BindingJsx, BindingLogLevel, BindingPluginOrder, BindingPropertyReadSideEffects, BindingPropertyWriteSideEffects, BindingWatcher, ParallelJsPluginRegistry, ResolverFactory, augmentCodeLocation, error, initTraceSubscriber, isolatedDeclaration, logCycleLoading, logDuplicateJsxConfig, logInputHookInOutputPlugin, logInvalidLogPosition, logMultiplyNotifyOption, logPluginError, moduleRunnerTransform, parseAst, parseAstAsync, registerPlugins, shutdownAsyncRuntime, startAsyncRuntime, transform };