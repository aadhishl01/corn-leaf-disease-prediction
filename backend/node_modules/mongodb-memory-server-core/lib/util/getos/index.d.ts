/** Helper Static so that a consistent UNKNOWN value is used */
export declare const UNKNOWN = "unknown";
export interface OtherOS {
    os: 'aix' | 'android' | 'darwin' | 'freebsd' | 'openbsd' | 'sunos' | 'win32' | 'cygwin' | string;
}
export interface LinuxOS extends OtherOS {
    os: 'linux';
    dist: string;
    release: string;
    codename?: string;
    id_like?: string[];
}
export type AnyOS = OtherOS | LinuxOS;
/**
 * Check if the OS is a LinuxOS Typeguard
 * @param os The OS object to check for
 */
export declare function isLinuxOS(os: AnyOS): os is LinuxOS;
/** Get an OS object */
export declare function getOS(): Promise<AnyOS>;
/**
 * Helper function to check if the input os is valid
 * @param os The OS information to check
 * @returns `true` if not undefined AND not UNKNOWN
 */
export declare function isValidOs(os: LinuxOS | undefined): os is LinuxOS;
/**
 * Parse LSB-like output (either command or file)
 */
export declare function parseLSB(input: string): LinuxOS;
/**
 * Parse OSRelease-like output
 */
export declare function parseOS(input: string): LinuxOS;
/**
 * Try to read a release file path and apply a parser to the output
 * @param path The Path to read for an release file
 * @param parser An function to parse the output of the file
 */
export declare function tryReleaseFile(path: string, parser: (output: string) => LinuxOS | undefined): Promise<LinuxOS | undefined>;
//# sourceMappingURL=index.d.ts.map