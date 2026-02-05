export class DevRuntime {
    /**
     * @param {WebSocket} socket
     */
    constructor(socket: WebSocket);
    socket: WebSocket;
    /**
     * @type {Record<string, Module>}
     */
    modules: Record<string, Module>;
    /**
     * @param {string} _moduleId
     */
    createModuleHotContext(_moduleId: string): void;
    /**
     * @param {string[]} _boundaries
     */
    applyUpdates(_boundaries: string[]): void;
    /**
     * @param {string} id
     * @param {{ exports: any }} exportsHolder
     */
    registerModule(id: string, exportsHolder: {
        exports: any;
    }): void;
    /**
     * @param {string} id
     */
    loadExports(id: string): any;
    /**
     * __esmMin
     *
     * @type {<T>(fn: any, res: T) => () => T}
     * @internal
     */
    createEsmInitializer: <T>(fn: any, res: T) => () => T;
    /**
     * __commonJSMin
     *
     * @type {<T extends { exports: any }>(cb: any, mod: { exports: any }) => () => T}
     * @internal
     */
    createCjsInitializer: <T extends {
        exports: any;
    }>(cb: any, mod: {
        exports: any;
    }) => () => T;
    /** @internal */
    __toESM: any;
    /** @internal */
    __toCommonJS: any;
    /** @internal */
    __export: any;
    /** @internal */
    __toDynamicImportESM: any;
    /** @internal */
    __reExport: any;
    sendModuleRegisteredMessage: (module: string) => void;
}
declare class Module {
    /**
     * @param {string} id
     */
    constructor(id: string);
    /**
     * @type {{ exports: any }}
     */
    exportsHolder: {
        exports: any;
    };
    /**
     * @type {string}
     */
    id: string;
    get exports(): any;
}
export {};
