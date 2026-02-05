import { BindingAssetPluginConfig, BindingBuildImportAnalysisPluginConfig, BindingClientHmrUpdate, BindingDynamicImportVarsPluginConfig, BindingEsmExternalRequirePluginConfig, BindingImportGlobPluginConfig, BindingIsolatedDeclarationPluginConfig, BindingJsonPluginConfig, BindingManifestPluginConfig, BindingModulePreloadPolyfillPluginConfig, BindingReactRefreshWrapperPluginConfig, BindingReplacePluginConfig, BindingReporterPluginConfig, BindingTransformPluginConfig, BindingViteResolvePluginConfig, BindingWasmHelperPluginConfig, IsolatedDeclarationsOptions, IsolatedDeclarationsResult, NapiResolveOptions, ResolveResult, ResolverFactory, TransformOptions, TransformResult, isolatedDeclaration, moduleRunnerTransform, transform } from "./shared/binding-DOpOtI1J.mjs";
import { BuiltinPlugin, InputOptions, OutputOptions, StringOrRegExp, defineParallelPlugin } from "./shared/define-config-BZ_n3PjJ.mjs";

//#region src/api/dev/dev-options.d.ts
interface DevWatchOptions {
  /**
  * If `true`, files are not written to disk.
  * @default false
  */
  skipWrite?: boolean;
  /**
  * If `true`, use polling instead of native file system events for watching.
  * @default false
  */
  usePolling?: boolean;
  /**
  * Poll interval in milliseconds (only used when usePolling is true).
  * @default 100
  */
  pollInterval?: number;
  /**
  * If `true`, use debounced watcher. If `false`, use non-debounced watcher for immediate responses.
  * @default true
  */
  useDebounce?: boolean;
  /**
  * Debounce duration in milliseconds (only used when useDebounce is true).
  * @default 10
  */
  debounceDuration?: number;
  /**
  * Whether to compare file contents for poll-based watchers (only used when usePolling is true).
  * When enabled, poll watchers will check file contents to determine if they actually changed.
  * @default false
  */
  compareContentsForPolling?: boolean;
  /**
  * Tick rate in milliseconds for debounced watchers (only used when useDebounce is true).
  * Controls how frequently the debouncer checks for events to process.
  * When not specified, the debouncer will auto-select an appropriate tick rate (1/4 of the debounce duration).
  * @default undefined (auto-select)
  */
  debounceTickRate?: number;
}
interface DevOptions {
  onHmrUpdates?: (updates: BindingClientHmrUpdate[], changedFiles: string[]) => void | Promise<void>;
  watch?: DevWatchOptions;
}
//#endregion
//#region src/api/dev/dev-engine.d.ts
declare class DevEngine {
  #private;
  static create(inputOptions: InputOptions, outputOptions?: OutputOptions, devOptions?: DevOptions): Promise<DevEngine>;
  private constructor();
  run(): Promise<void>;
  ensureCurrentBuildFinish(): Promise<void>;
  hasLatestBuildOutput(): Promise<boolean>;
  ensureLatestBuildOutput(): Promise<void>;
  invalidate(file: string, firstInvalidatedBy?: string): Promise<BindingClientHmrUpdate[]>;
  registerModules(clientId: string, modules: string[]): void;
  removeClient(clientId: string): void;
}
//#endregion
//#region src/api/dev/index.d.ts
declare var dev: typeof DevEngine.create;
//#endregion
//#region src/api/experimental.d.ts
/**
* This is an experimental API. It's behavior may change in the future.
*
* Calling this API will only execute the scan stage of rolldown.
*/
declare const scan: (input: InputOptions) => Promise<void>;
//#endregion
//#region src/builtin-plugin/constructors.d.ts
declare function modulePreloadPolyfillPlugin(config?: BindingModulePreloadPolyfillPluginConfig): BuiltinPlugin;
type DynamicImportVarsPluginConfig = Omit<BindingDynamicImportVarsPluginConfig, "include" | "exclude"> & {
  include?: StringOrRegExp | StringOrRegExp[];
  exclude?: StringOrRegExp | StringOrRegExp[];
};
declare function dynamicImportVarsPlugin(config?: DynamicImportVarsPluginConfig): BuiltinPlugin;
declare function importGlobPlugin(config?: BindingImportGlobPluginConfig): BuiltinPlugin;
declare function reporterPlugin(config?: BindingReporterPluginConfig): BuiltinPlugin;
declare function manifestPlugin(config?: BindingManifestPluginConfig): BuiltinPlugin;
declare function wasmHelperPlugin(config?: BindingWasmHelperPluginConfig): BuiltinPlugin;
declare function wasmFallbackPlugin(): BuiltinPlugin;
declare function loadFallbackPlugin(): BuiltinPlugin;
declare function jsonPlugin(config?: BindingJsonPluginConfig): BuiltinPlugin;
declare function buildImportAnalysisPlugin(config: BindingBuildImportAnalysisPluginConfig): BuiltinPlugin;
declare function viteResolvePlugin(config: BindingViteResolvePluginConfig): BuiltinPlugin;
declare function isolatedDeclarationPlugin(config?: BindingIsolatedDeclarationPluginConfig): BuiltinPlugin;
declare function assetPlugin(config?: BindingAssetPluginConfig): BuiltinPlugin;
declare function webWorkerPostPlugin(): BuiltinPlugin;
declare function esmExternalRequirePlugin(config?: BindingEsmExternalRequirePluginConfig): BuiltinPlugin;
type ReactRefreshWrapperPluginConfig = Omit<BindingReactRefreshWrapperPluginConfig, "include" | "exclude"> & {
  include?: StringOrRegExp | StringOrRegExp[];
  exclude?: StringOrRegExp | StringOrRegExp[];
};
declare function reactRefreshWrapperPlugin(config: ReactRefreshWrapperPluginConfig): BuiltinPlugin;
//#endregion
//#region src/builtin-plugin/alias-plugin.d.ts
type AliasPluginAlias = {
  find: string | RegExp;
  replacement: string;
};
type AliasPluginConfig = {
  entries: AliasPluginAlias[];
};
declare function aliasPlugin(config: AliasPluginConfig): BuiltinPlugin;
//#endregion
//#region src/builtin-plugin/replace-plugin.d.ts
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
declare function replacePlugin(values?: BindingReplacePluginConfig["values"], options?: Omit<BindingReplacePluginConfig, "values">): BuiltinPlugin;
//#endregion
//#region src/builtin-plugin/transform-plugin.d.ts
type TransformPattern = string | RegExp | readonly (RegExp | string)[];
type TransformPluginConfig = Omit<BindingTransformPluginConfig, "include" | "exclude" | "jsxRefreshInclude" | "jsxRefreshExclude"> & {
  include?: TransformPattern;
  exclude?: TransformPattern;
  jsxRefreshInclude?: TransformPattern;
  jsxRefreshExclude?: TransformPattern;
};
declare function transformPlugin(config?: TransformPluginConfig): BuiltinPlugin;
//#endregion
export { BindingClientHmrUpdate, DevEngine, type DevOptions, type DevWatchOptions, type IsolatedDeclarationsOptions, type IsolatedDeclarationsResult, type NapiResolveOptions as ResolveOptions, type ResolveResult, ResolverFactory, type TransformOptions, type TransformResult, aliasPlugin, assetPlugin, buildImportAnalysisPlugin, defineParallelPlugin, dev, dynamicImportVarsPlugin, esmExternalRequirePlugin, importGlobPlugin, isolatedDeclaration, isolatedDeclarationPlugin, jsonPlugin, loadFallbackPlugin, manifestPlugin, modulePreloadPolyfillPlugin, moduleRunnerTransform, reactRefreshWrapperPlugin, replacePlugin, reporterPlugin, scan, transform, transformPlugin, viteResolvePlugin, wasmFallbackPlugin, wasmHelperPlugin, webWorkerPostPlugin };