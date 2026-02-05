import "./shared/binding-DOpOtI1J.mjs";
import { ConfigExport, defineConfig } from "./shared/define-config-BZ_n3PjJ.mjs";

//#region src/utils/load-config.d.ts
declare function loadConfig(configPath: string): Promise<ConfigExport>;
//#endregion
//#region src/config.d.ts
declare const VERSION: string;
//#endregion
export { VERSION, defineConfig, loadConfig };