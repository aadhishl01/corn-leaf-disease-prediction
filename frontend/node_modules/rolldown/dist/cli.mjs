import "./shared/parse-ast-index-D2PcAmXE.mjs";
import { description, getCliSchemaInfo, getInputCliKeys, getOutputCliKeys, onExit, rolldown, validateCliOptions, version, watch } from "./shared/src-DkvlJJsC.mjs";
import { arraify } from "./shared/misc-CQeo-AFx.mjs";
import { logger } from "./shared/logger-ClMekpHZ.mjs";
import { loadConfig } from "./shared/load-config-DkgIoWHl.mjs";
import path from "node:path";
import colors from "ansis";
import { parseArgs } from "node:util";
import process$1 from "node:process";
import { performance } from "node:perf_hooks";

//#region src/cli/arguments/alias.ts
const alias = {
	config: {
		abbreviation: "c",
		hint: "filename"
	},
	help: { abbreviation: "h" },
	version: { abbreviation: "v" },
	watch: { abbreviation: "w" },
	dir: { abbreviation: "d" },
	file: { abbreviation: "o" },
	external: { abbreviation: "e" },
	format: { abbreviation: "f" },
	name: { abbreviation: "n" },
	globals: { abbreviation: "g" },
	sourcemap: {
		abbreviation: "s",
		default: true
	},
	minify: { abbreviation: "m" },
	platform: { abbreviation: "p" },
	assetFileNames: { hint: "name" },
	chunkFileNames: { hint: "name" },
	entryFileNames: { hint: "name" },
	externalLiveBindings: {
		default: true,
		reverse: true
	},
	treeshake: {
		default: true,
		reverse: true
	},
	preserveEntrySignatures: {
		default: "strict",
		reverse: true
	},
	moduleTypes: { hint: "types" }
};

//#endregion
//#region src/cli/arguments/utils.ts
function setNestedProperty(obj, path$1, value) {
	const keys = path$1.split(".");
	let current = obj;
	for (let i = 0; i < keys.length - 1; i++) {
		if (!current[keys[i]]) current[keys[i]] = {};
		current = current[keys[i]];
	}
	const finalKey = keys[keys.length - 1];
	Object.defineProperty(current, finalKey, {
		value,
		writable: true,
		enumerable: true,
		configurable: true
	});
}
function camelCaseToKebabCase(str) {
	return str.replace(/[A-Z]/g, (match) => `-${match.toLowerCase()}`);
}
function kebabCaseToCamelCase(str) {
	return str.replace(/-./g, (match) => match[1].toUpperCase());
}

//#endregion
//#region src/cli/arguments/normalize.ts
function normalizeCliOptions(cliOptions, positionals) {
	const [data, errors] = validateCliOptions(cliOptions);
	if (errors?.length) {
		errors.forEach((error) => {
			logger.error(`${error}. You can use \`rolldown -h\` to see the help.`);
		});
		process.exit(1);
	}
	const options$1 = data ?? {};
	const result = {
		input: {},
		output: {},
		help: options$1.help ?? false,
		version: options$1.version ?? false,
		watch: options$1.watch ?? false
	};
	if (typeof options$1.config === "string") result.config = options$1.config;
	const keysOfInput = getInputCliKeys();
	const keysOfOutput = getOutputCliKeys();
	const reservedKeys = [
		"help",
		"version",
		"config",
		"watch"
	];
	for (let [key, value] of Object.entries(options$1)) {
		const [primary] = key.split(".");
		if (keysOfInput.includes(primary)) setNestedProperty(result.input, key, value);
		else if (keysOfOutput.includes(primary)) setNestedProperty(result.output, key, value);
		else if (!reservedKeys.includes(key)) {
			logger.error(`Unknown option: ${key}`);
			process.exit(1);
		}
	}
	if (!result.config && positionals.length > 0) if (Array.isArray(result.input.input)) result.input.input.push(...positionals);
	else result.input.input = positionals;
	return result;
}

//#endregion
//#region src/cli/arguments/index.ts
const schemaInfo = getCliSchemaInfo();
const options = Object.fromEntries(Object.entries(schemaInfo).filter(([_key, info]) => info.type !== "never").map(([key, info]) => {
	const config = Object.getOwnPropertyDescriptor(alias, key)?.value;
	const result = {
		type: info.type === "boolean" ? "boolean" : "string",
		description: info?.description ?? config?.description ?? "",
		hint: config?.hint
	};
	if (config && config?.abbreviation) result.short = config?.abbreviation;
	if (config && config.reverse) {
		if (result.description.startsWith("enable")) result.description = result.description.replace("enable", "disable");
		else if (!result.description.startsWith("Avoid")) result.description = `disable ${result.description}`;
	}
	key = camelCaseToKebabCase(key);
	return [config?.reverse ? `no-${key}` : key, result];
}));
function parseCliArguments() {
	const { values, tokens, positionals } = parseArgs({
		options,
		tokens: true,
		allowPositionals: true,
		strict: false
	});
	let invalid_options = tokens.filter((token) => token.kind === "option").map((option) => {
		let negative = false;
		if (option.name.startsWith("no-")) {
			const name = kebabCaseToCamelCase(option.name.substring(3));
			if (name in schemaInfo) {
				delete values[option.name];
				option.name = name;
				negative = true;
			}
		}
		delete values[option.name];
		option.name = kebabCaseToCamelCase(option.name);
		let originalInfo = schemaInfo[option.name];
		if (!originalInfo) return {
			name: option.name,
			value: option.value
		};
		let type = originalInfo.type;
		if (type === "string" && typeof option.value !== "string") {
			let opt = option;
			let defaultValue = Object.getOwnPropertyDescriptor(alias, opt.name)?.value;
			Object.defineProperty(values, opt.name, {
				value: defaultValue.default ?? "",
				enumerable: true,
				configurable: true,
				writable: true
			});
		} else if (type === "object" && typeof option.value === "string") {
			const [key, value] = option.value.split(",").map((x) => x.split("="))[0];
			if (!values[option.name]) Object.defineProperty(values, option.name, {
				value: {},
				enumerable: true,
				configurable: true,
				writable: true
			});
			if (key && value) Object.defineProperty(values[option.name], key, {
				value,
				enumerable: true,
				configurable: true,
				writable: true
			});
		} else if (type === "array" && typeof option.value === "string") {
			if (!values[option.name]) Object.defineProperty(values, option.name, {
				value: [],
				enumerable: true,
				configurable: true,
				writable: true
			});
			values[option.name].push(option.value);
		} else if (type === "boolean") Object.defineProperty(values, option.name, {
			value: !negative,
			enumerable: true,
			configurable: true,
			writable: true
		});
		else if (type === "union") {
			let defaultValue = Object.getOwnPropertyDescriptor(alias, option.name)?.value;
			Object.defineProperty(values, option.name, {
				value: option.value ?? defaultValue?.default ?? "",
				enumerable: true,
				configurable: true,
				writable: true
			});
		} else Object.defineProperty(values, option.name, {
			value: option.value ?? "",
			enumerable: true,
			configurable: true,
			writable: true
		});
	}).filter((item) => {
		return item !== void 0;
	});
	invalid_options.sort((a, b) => {
		return a.name.localeCompare(b.name);
	});
	if (invalid_options.length !== 0) {
		let single = invalid_options.length === 1;
		logger.warn(`Option \`${invalid_options.map((item) => item.name).join(",")}\` ${single ? "is" : "are"} unrecognized. We will ignore ${single ? "this" : "those"} option${single ? "" : "s"}.`);
	}
	let rawArgs = {
		...values,
		...invalid_options.reduce((acc, cur) => {
			acc[cur.name] = cur.value;
			return acc;
		}, Object.create(null))
	};
	return {
		...normalizeCliOptions(values, positionals),
		rawArgs
	};
}

//#endregion
//#region src/utils/clear-screen.ts
const CLEAR_SCREEN = "\x1Bc";
function getClearScreenFunction(options$1) {
	const isTTY = process.stdout.isTTY;
	const isAnyOptionNotAllowingClearScreen = arraify(options$1).some(({ watch: watch$1 }) => watch$1 === false || watch$1?.clearScreen === false);
	if (isTTY && !isAnyOptionNotAllowingClearScreen) return () => {
		process.stdout.write(CLEAR_SCREEN);
	};
}

//#endregion
//#region \0@oxc-project+runtime@0.93.0/helpers/usingCtx.js
function _usingCtx() {
	var r = "function" == typeof SuppressedError ? SuppressedError : function(r$1, e$1) {
		var n$1 = Error();
		return n$1.name = "SuppressedError", n$1.error = r$1, n$1.suppressed = e$1, n$1;
	}, e = {}, n = [];
	function using(r$1, e$1) {
		if (null != e$1) {
			if (Object(e$1) !== e$1) throw new TypeError("using declarations can only be used with objects, functions, null, or undefined.");
			if (r$1) var o = e$1[Symbol.asyncDispose || Symbol["for"]("Symbol.asyncDispose")];
			if (void 0 === o && (o = e$1[Symbol.dispose || Symbol["for"]("Symbol.dispose")], r$1)) var t = o;
			if ("function" != typeof o) throw new TypeError("Object is not disposable.");
			t && (o = function o$1() {
				try {
					t.call(e$1);
				} catch (r$2) {
					return Promise.reject(r$2);
				}
			}), n.push({
				v: e$1,
				d: o,
				a: r$1
			});
		} else r$1 && n.push({
			d: e$1,
			a: r$1
		});
		return e$1;
	}
	return {
		e,
		u: using.bind(null, !1),
		a: using.bind(null, !0),
		d: function d() {
			var o, t = this.e, s = 0;
			function next() {
				for (; o = n.pop();) try {
					if (!o.a && 1 === s) return s = 0, n.push(o), Promise.resolve().then(next);
					if (o.d) {
						var r$1 = o.d.call(o.v);
						if (o.a) return s |= 2, Promise.resolve(r$1).then(next, err);
					} else s |= 1;
				} catch (r$2) {
					return err(r$2);
				}
				if (1 === s) return t !== e ? Promise.reject(t) : Promise.resolve();
				if (t !== e) throw t;
			}
			function err(n$1) {
				return t = t !== e ? new r(n$1, t) : n$1, next();
			}
			return next();
		}
	};
}

//#endregion
//#region src/cli/commands/bundle.ts
async function bundleWithConfig(configPath, cliOptions, rawArgs = {}) {
	if (cliOptions.watch) {
		process.env.ROLLUP_WATCH = "true";
		process.env.ROLLDOWN_WATCH = "true";
	}
	const config = await loadConfig(configPath);
	if (!config) {
		logger.error(`No configuration found at ${config}`);
		process.exit(1);
	}
	const resolvedConfig = typeof config === "function" ? await config(rawArgs) : config;
	if (cliOptions.watch) await watchInner(resolvedConfig, cliOptions);
	else await bundleInner(resolvedConfig, cliOptions);
}
async function bundleWithCliOptions(cliOptions) {
	try {
		var _usingCtx$1 = _usingCtx();
		if (cliOptions.output.dir || cliOptions.output.file) {
			await (cliOptions.watch ? watchInner : bundleInner)({}, cliOptions);
			return;
		}
		if (cliOptions.watch) {
			logger.error("You must specify `output.dir` to use watch mode");
			process.exit(1);
		}
		const { output: outputs } = await _usingCtx$1.a(await rolldown(cliOptions.input)).generate(cliOptions.output);
		if (outputs.length === 0) {
			logger.error("No output generated");
			process.exit(1);
		}
		for (const file of outputs) {
			if (outputs.length > 1) logger.log(`\n${colors.cyan(colors.bold(`|→ ${file.fileName}:`))}\n`);
			console.log(file.type === "asset" ? file.source : file.code);
		}
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		await _usingCtx$1.d();
	}
}
async function watchInner(config, cliOptions) {
	let normalizedConfig = arraify(config).map((option) => {
		return {
			...option,
			...cliOptions.input,
			output: arraify(option.output || {}).map((output) => {
				return {
					...output,
					...cliOptions.output
				};
			})
		};
	});
	const watcher = watch(normalizedConfig);
	onExit((code) => {
		Promise.resolve(watcher.close()).finally(() => {
			process.exit(typeof code === "number" ? code : 0);
		});
		return true;
	});
	const changedFile = [];
	watcher.on("change", (id, event) => {
		if (event.event === "update") changedFile.push(id);
	});
	const clearScreen = getClearScreenFunction(normalizedConfig);
	watcher.on("event", async (event) => {
		switch (event.code) {
			case "START":
				clearScreen?.();
				break;
			case "BUNDLE_START":
				if (changedFile.length > 0) logger.log(`Found ${colors.bold(changedFile.map(relativeId).join(", "))} changed, rebuilding...`);
				changedFile.length = 0;
				break;
			case "BUNDLE_END":
				await event.result.close();
				logger.success(`Rebuilt ${colors.bold(relativeId(event.output[0]))} in ${colors.green(ms(event.duration))}.`);
				break;
			case "ERROR":
				await event.result.close();
				logger.error(event.error);
				break;
			default: break;
		}
	});
	logger.log(`Waiting for changes...`);
}
async function bundleInner(config, cliOptions) {
	const startTime = performance.now();
	const result = [];
	const configList = arraify(config);
	for (const config$1 of configList) {
		const outputList = arraify(config$1.output || {});
		const build = await rolldown({
			...config$1,
			...cliOptions.input
		});
		for (const output of outputList) try {
			result.push(await build.write({
				...output,
				...cliOptions.output
			}));
		} finally {
			await build.close();
		}
	}
	result.forEach(printBundleOutputPretty);
	logger.log(``);
	const duration = performance.now() - startTime;
	logger.success(`rolldown v${version} Finished in ${colors.green(ms(duration))}`);
}
function printBundleOutputPretty(output) {
	const outputEntries = collectOutputEntries(output.output);
	const outputLayoutSizes = collectOutputLayoutAdjustmentSizes(outputEntries);
	printOutputEntries(outputEntries, outputLayoutSizes, "<DIR>");
}
function collectOutputEntries(output) {
	return output.map((chunk) => ({
		type: chunk.type,
		fileName: chunk.fileName,
		size: chunk.type === "chunk" ? Buffer.byteLength(chunk.code) : Buffer.byteLength(chunk.source)
	}));
}
function collectOutputLayoutAdjustmentSizes(entries) {
	let longest = 0;
	let biggestSize = 0;
	for (const entry of entries) {
		if (entry.fileName.length > longest) longest = entry.fileName.length;
		if (entry.size > biggestSize) biggestSize = entry.size;
	}
	const sizePad = displaySize(biggestSize).length;
	return {
		longest,
		biggestSize,
		sizePad
	};
}
const numberFormatter = new Intl.NumberFormat("en", {
	maximumFractionDigits: 2,
	minimumFractionDigits: 2
});
function displaySize(bytes) {
	return `${numberFormatter.format(bytes / 1e3)} kB`;
}
const CHUNK_GROUPS = [{
	type: "asset",
	color: "green"
}, {
	type: "chunk",
	color: "cyan"
}];
function printOutputEntries(entries, sizeAdjustment, distPath) {
	for (const group of CHUNK_GROUPS) {
		const filtered = entries.filter((e) => e.type === group.type);
		if (!filtered.length) continue;
		for (const entry of filtered.sort((a, z) => a.size - z.size)) {
			let log = colors.dim(withTrailingSlash(distPath));
			log += colors[group.color](entry.fileName.padEnd(sizeAdjustment.longest + 2));
			log += colors.dim(entry.type);
			log += colors.dim(` │ size: ${displaySize(entry.size).padStart(sizeAdjustment.sizePad)}`);
			logger.log(log);
		}
	}
}
function withTrailingSlash(path$1) {
	if (path$1[path$1.length - 1] !== "/") return `${path$1}/`;
	return path$1;
}
function ms(duration) {
	return duration < 1e3 ? `${duration.toFixed(2)} ms` : `${(duration / 1e3).toFixed(2)} s`;
}
function relativeId(id) {
	if (!path.isAbsolute(id)) return id;
	return path.relative(path.resolve(), id);
}

//#endregion
//#region src/cli/commands/help.ts
const introduction = `${colors.gray(`${description} (rolldown v${version})`)}

${colors.bold(colors.underline("USAGE"))} ${colors.cyan("rolldown -c <config>")} or ${colors.cyan("rolldown <input> <options>")}`;
const examples = [
	{
		title: "Bundle with a config file `rolldown.config.mjs`",
		command: "rolldown -c rolldown.config.mjs"
	},
	{
		title: "Bundle the `src/main.ts` to `dist` with `cjs` format",
		command: "rolldown src/main.ts -d dist -f cjs"
	},
	{
		title: "Bundle the `src/main.ts` and handle the `.png` assets to Data URL",
		command: "rolldown src/main.ts -d dist --moduleTypes .png=dataurl"
	},
	{
		title: "Bundle the `src/main.tsx` and minify the output with sourcemap",
		command: "rolldown src/main.tsx -d dist -m -s"
	},
	{
		title: "Create self-executing IIFE using external jQuery as `$` and `_`",
		command: "rolldown src/main.ts -d dist -n bundle -f iife -e jQuery,window._ -g jQuery=$"
	}
];
const notes = [
	"Due to the API limitation, you need to pass `-s` for `.map` sourcemap file as the last argument.",
	"If you are using the configuration, please pass the `-c` as the last argument if you ignore the default configuration file.",
	"CLI options will override the configuration file.",
	"For more information, please visit https://rolldown.rs/."
];
function showHelp() {
	logger.log(introduction);
	logger.log("");
	logger.log(`${colors.bold(colors.underline("OPTIONS"))}`);
	logger.log("");
	logger.log(Object.entries(options).sort(([a], [b]) => {
		if (options[a].short && !options[b].short) return -1;
		if (!options[a].short && options[b].short) return 1;
		if (options[a].short && options[b].short) return options[a].short.localeCompare(options[b].short);
		return a.localeCompare(b);
	}).map(([option, { type, short, hint, description: description$1 }]) => {
		let optionStr = `  --${option} `;
		option = camelCaseToKebabCase(option);
		if (short) optionStr += `-${short}, `;
		if (type === "string") optionStr += `<${hint ?? option}>`;
		if (description$1 && description$1.length > 0) description$1 = description$1[0].toUpperCase() + description$1.slice(1);
		return colors.cyan(optionStr.padEnd(30)) + description$1 + (description$1 && description$1?.endsWith(".") ? "" : ".");
	}).join("\n"));
	logger.log("");
	logger.log(`${colors.bold(colors.underline("EXAMPLES"))}`);
	logger.log("");
	examples.forEach(({ title, command }, ord) => {
		logger.log(`  ${ord + 1}. ${title}:`);
		logger.log(`    ${colors.cyan(command)}`);
		logger.log("");
	});
	logger.log(`${colors.bold(colors.underline("NOTES"))}`);
	logger.log("");
	notes.forEach((note) => {
		logger.log(`  * ${colors.gray(note)}`);
	});
}

//#endregion
//#region src/cli/version-check.ts
function checkNodeVersion(nodeVersion) {
	const currentVersion = nodeVersion.split(".");
	const major = parseInt(currentVersion[0], 10);
	const minor = parseInt(currentVersion[1], 10);
	return major === 20 && minor >= 19 || major === 22 && minor >= 12 || major > 22;
}

//#endregion
//#region src/cli/index.ts
if (!checkNodeVersion(process$1.versions.node)) logger.warn(`You are using Node.js ${process$1.versions.node}. Rolldown requires Node.js version 20.19+ or 22.12+. Please upgrade your Node.js version.`);
async function main() {
	const { rawArgs,...cliOptions } = parseCliArguments();
	if (cliOptions.config || cliOptions.config === "") {
		await bundleWithConfig(cliOptions.config, cliOptions, rawArgs);
		return;
	}
	if ("input" in cliOptions.input) {
		await bundleWithCliOptions(cliOptions);
		return;
	}
	if (cliOptions.version) {
		logger.log(`rolldown v${version}`);
		return;
	}
	showHelp();
}
main().catch((err) => {
	logger.error(err);
	process$1.exit(1);
});

//#endregion
export {  };