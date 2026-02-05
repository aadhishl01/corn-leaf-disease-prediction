//#region src/utils/misc.ts
function arraify(value) {
	return Array.isArray(value) ? value : [value];
}
function isPromiseLike(value) {
	return value && (typeof value === "object" || typeof value === "function") && typeof value.then === "function";
}
function unimplemented(info) {
	if (info) throw new Error(`unimplemented: ${info}`);
	throw new Error("unimplemented");
}
function unreachable(info) {
	if (info) throw new Error(`unreachable: ${info}`);
	throw new Error("unreachable");
}
function unsupported(info) {
	throw new Error(`UNSUPPORTED: ${info}`);
}
function noop(..._args) {}

//#endregion
export { arraify, isPromiseLike, noop, unimplemented, unreachable, unsupported };