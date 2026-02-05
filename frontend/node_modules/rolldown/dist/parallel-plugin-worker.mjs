import { registerPlugins } from "./shared/parse-ast-index-D2PcAmXE.mjs";
import { PluginContextData, bindingifyPlugin } from "./shared/src-DkvlJJsC.mjs";
import "./shared/misc-CQeo-AFx.mjs";
import { parentPort, workerData } from "node:worker_threads";

//#region src/parallel-plugin-worker.ts
const { registryId, pluginInfos, threadNumber } = workerData;
(async () => {
	try {
		const plugins = await Promise.all(pluginInfos.map(async (pluginInfo) => {
			const definePluginImpl = (await import(pluginInfo.fileUrl)).default;
			const plugin = await definePluginImpl(pluginInfo.options, { threadNumber });
			return {
				index: pluginInfo.index,
				plugin: bindingifyPlugin(plugin, {}, {}, new PluginContextData(() => {}, {}, []), [], () => {}, "info", false)
			};
		}));
		registerPlugins(registryId, plugins);
		parentPort.postMessage({ type: "success" });
	} catch (error) {
		parentPort.postMessage({
			type: "error",
			error
		});
	} finally {
		parentPort.unref();
	}
})();

//#endregion
export {  };