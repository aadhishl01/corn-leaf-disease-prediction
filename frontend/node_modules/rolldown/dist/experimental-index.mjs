import { BindingClientHmrUpdate, BindingDevEngine, ResolverFactory, isolatedDeclaration, moduleRunnerTransform, transform } from "./shared/parse-ast-index-D2PcAmXE.mjs";
import { BuiltinPlugin, PluginDriver, RolldownBuild, createBundlerOptions, makeBuiltinPluginCallable, normalizedStringOrRegex } from "./shared/src-DkvlJJsC.mjs";
import "./shared/misc-CQeo-AFx.mjs";
import { logger } from "./shared/logger-ClMekpHZ.mjs";
import { pathToFileURL } from "node:url";

//#region src/api/dev/dev-engine.ts
var DevEngine = class DevEngine {
	#inner;
	#cachedBuildFinishPromise = null;
	static async create(inputOptions, outputOptions = {}, devOptions = {}) {
		inputOptions = await PluginDriver.callOptionsHook(inputOptions);
		const options = await createBundlerOptions(inputOptions, outputOptions, false);
		const bindingDevOptions = {
			onHmrUpdates: devOptions.onHmrUpdates,
			watch: devOptions.watch && {
				skipWrite: devOptions.watch.skipWrite,
				usePolling: devOptions.watch.usePolling,
				pollInterval: devOptions.watch.pollInterval,
				useDebounce: devOptions.watch.useDebounce,
				debounceDuration: devOptions.watch.debounceDuration,
				compareContentsForPolling: devOptions.watch.compareContentsForPolling,
				debounceTickRate: devOptions.watch.debounceTickRate
			}
		};
		const inner = new BindingDevEngine(options.bundlerOptions, bindingDevOptions);
		return new DevEngine(inner);
	}
	constructor(inner) {
		this.#inner = inner;
	}
	async run() {
		await this.#inner.run();
	}
	async ensureCurrentBuildFinish() {
		if (this.#cachedBuildFinishPromise) return this.#cachedBuildFinishPromise;
		const promise = this.#inner.ensureCurrentBuildFinish().then(() => {
			this.#cachedBuildFinishPromise = null;
		});
		this.#cachedBuildFinishPromise = promise;
		return promise;
	}
	async hasLatestBuildOutput() {
		return this.#inner.hasLatestBuildOutput();
	}
	async ensureLatestBuildOutput() {
		await this.#inner.ensureLatestBuildOutput();
	}
	async invalidate(file, firstInvalidatedBy) {
		return this.#inner.invalidate(file, firstInvalidatedBy);
	}
	registerModules(clientId, modules) {
		this.#inner.registerModules(clientId, modules);
	}
	removeClient(clientId) {
		this.#inner.removeClient(clientId);
	}
};

//#endregion
//#region src/api/dev/index.ts
var dev = DevEngine.create;

//#endregion
//#region src/api/experimental.ts
/**
* This is an experimental API. It's behavior may change in the future.
*
* Calling this API will only execute the scan stage of rolldown.
*/
const scan = async (input) => {
	const inputOptions = await PluginDriver.callOptionsHook(input);
	const build = new RolldownBuild(inputOptions);
	try {
		await build.scan();
	} finally {
		await build.close();
	}
};

//#endregion
//#region src/plugin/parallel-plugin.ts
function defineParallelPlugin(pluginPath) {
	return (options) => {
		return { _parallel: {
			fileUrl: pathToFileURL(pluginPath).href,
			options
		} };
	};
}

//#endregion
//#region src/builtin-plugin/constructors.ts
function modulePreloadPolyfillPlugin(config) {
	return new BuiltinPlugin("builtin:module-preload-polyfill", config);
}
function dynamicImportVarsPlugin(config) {
	if (config) {
		config.include = normalizedStringOrRegex(config.include);
		config.exclude = normalizedStringOrRegex(config.exclude);
	}
	return new BuiltinPlugin("builtin:dynamic-import-vars", config);
}
function importGlobPlugin(config) {
	return new BuiltinPlugin("builtin:import-glob", config);
}
function reporterPlugin(config) {
	return new BuiltinPlugin("builtin:reporter", config);
}
function manifestPlugin(config) {
	return new BuiltinPlugin("builtin:manifest", config);
}
function wasmHelperPlugin(config) {
	return new BuiltinPlugin("builtin:wasm-helper", config);
}
function wasmFallbackPlugin() {
	const builtinPlugin = new BuiltinPlugin("builtin:wasm-fallback");
	return makeBuiltinPluginCallable(builtinPlugin);
}
function loadFallbackPlugin() {
	return new BuiltinPlugin("builtin:load-fallback");
}
function jsonPlugin(config) {
	const builtinPlugin = new BuiltinPlugin("builtin:json", config);
	return makeBuiltinPluginCallable(builtinPlugin);
}
function buildImportAnalysisPlugin(config) {
	return new BuiltinPlugin("builtin:build-import-analysis", config);
}
function viteResolvePlugin(config) {
	const builtinPlugin = new BuiltinPlugin("builtin:vite-resolve", config);
	return makeBuiltinPluginCallable(builtinPlugin);
}
function isolatedDeclarationPlugin(config) {
	return new BuiltinPlugin("builtin:isolated-declaration", config);
}
function assetPlugin(config) {
	return new BuiltinPlugin("builtin:asset", config);
}
function webWorkerPostPlugin() {
	return new BuiltinPlugin("builtin:web-worker-post");
}
function esmExternalRequirePlugin(config) {
	return new BuiltinPlugin("builtin:esm-external-require", config);
}
function reactRefreshWrapperPlugin(config) {
	if (config) {
		config.include = normalizedStringOrRegex(config.include);
		config.exclude = normalizedStringOrRegex(config.exclude);
	}
	const builtinPlugin = new BuiltinPlugin("builtin:react-refresh-wrapper", config);
	return makeBuiltinPluginCallable(builtinPlugin);
}

//#endregion
//#region src/builtin-plugin/alias-plugin.ts
function aliasPlugin(config) {
	return new BuiltinPlugin("builtin:alias", config);
}

//#endregion
//#region src/builtin-plugin/replace-plugin.ts
/**
* Replaces targeted strings in files while bundling.
*
* @example
* // Basic usage
* ```js
* replacePlugin({
*   'process.env.NODE_ENV': JSON.stringify('production'),
*    __buildVersion: 15
* })
* ```
* @example
* // With options
* ```js
* replacePlugin({
*   'process.env.NODE_ENV': JSON.stringify('production'),
*   __buildVersion: 15
* }, {
*   preventAssignment: false,
* })
* ```
*/
function replacePlugin(values = {}, options = {}) {
	let hasNonStringValues = false;
	Object.keys(values).forEach((key) => {
		const value = values[key];
		if (typeof value !== "string") {
			hasNonStringValues = true;
			values[key] = String(value);
		}
	});
	if (hasNonStringValues) logger.warn("Some values provided to `replacePlugin` are not strings. They will be converted to strings, but for better performance consider converting them manually.");
	return new BuiltinPlugin("builtin:replace", {
		...options,
		values
	});
}

//#endregion
//#region src/builtin-plugin/transform-plugin.ts
function transformPlugin(config) {
	if (config) config = {
		...config,
		include: normalizedStringOrRegex(config.include),
		exclude: normalizedStringOrRegex(config.exclude),
		jsxRefreshInclude: normalizedStringOrRegex(config.jsxRefreshInclude),
		jsxRefreshExclude: normalizedStringOrRegex(config.jsxRefreshExclude)
	};
	return new BuiltinPlugin("builtin:transform", config);
}

//#endregion
export { BindingClientHmrUpdate, DevEngine, ResolverFactory, aliasPlugin, assetPlugin, buildImportAnalysisPlugin, defineParallelPlugin, dev, dynamicImportVarsPlugin, esmExternalRequirePlugin, importGlobPlugin, isolatedDeclaration, isolatedDeclarationPlugin, jsonPlugin, loadFallbackPlugin, manifestPlugin, modulePreloadPolyfillPlugin, moduleRunnerTransform, reactRefreshWrapperPlugin, replacePlugin, reporterPlugin, scan, transform, transformPlugin, viteResolvePlugin, wasmFallbackPlugin, wasmHelperPlugin, webWorkerPostPlugin };