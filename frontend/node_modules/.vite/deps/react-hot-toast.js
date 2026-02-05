"use client";


import { __toESM, require_react } from "./react-CSb6HjG4.js";

//#region node_modules/goober/dist/goober.modern.js
var import_react = /* @__PURE__ */ __toESM(require_react(), 1);
var e = { data: "" }, t = (t$1) => {
	if ("object" == typeof window) {
		let e$1 = (t$1 ? t$1.querySelector("#_goober") : window._goober) || Object.assign(document.createElement("style"), {
			innerHTML: " ",
			id: "_goober"
		});
		return e$1.nonce = window.__nonce__, e$1.parentNode || (t$1 || document.head).appendChild(e$1), e$1.firstChild;
	}
	return t$1 || e;
}, r = (e$1) => {
	let r$1 = t(e$1), l$1 = r$1.data;
	return r$1.data = "", l$1;
}, l = /(?:([\u0080-\uFFFF\w-%@]+) *:? *([^{;]+?);|([^;}{]*?) *{)|(}\s*)/g, a = /\/\*[^]*?\*\/|  +/g, n$1 = /\n+/g, o = (e$1, t$1) => {
	let r$1 = "", l$1 = "", a$1 = "";
	for (let n$2 in e$1) {
		let c$1 = e$1[n$2];
		"@" == n$2[0] ? "i" == n$2[1] ? r$1 = n$2 + " " + c$1 + ";" : l$1 += "f" == n$2[1] ? o(c$1, n$2) : n$2 + "{" + o(c$1, "k" == n$2[1] ? "" : t$1) + "}" : "object" == typeof c$1 ? l$1 += o(c$1, t$1 ? t$1.replace(/([^,])+/g, (e$2) => n$2.replace(/([^,]*:\S+\([^)]*\))|([^,])+/g, (t$2) => /&/.test(t$2) ? t$2.replace(/&/g, e$2) : e$2 ? e$2 + " " + t$2 : t$2)) : n$2) : null != c$1 && (n$2 = /^--/.test(n$2) ? n$2 : n$2.replace(/[A-Z]/g, "-$&").toLowerCase(), a$1 += o.p ? o.p(n$2, c$1) : n$2 + ":" + c$1 + ";");
	}
	return r$1 + (t$1 && a$1 ? t$1 + "{" + a$1 + "}" : a$1) + l$1;
}, c = {}, s = (e$1) => {
	if ("object" == typeof e$1) {
		let t$1 = "";
		for (let r$1 in e$1) t$1 += r$1 + s(e$1[r$1]);
		return t$1;
	}
	return e$1;
}, i = (e$1, t$1, r$1, i$1, p$1) => {
	let u$1 = s(e$1), d$1 = c[u$1] || (c[u$1] = ((e$2) => {
		let t$2 = 0, r$2 = 11;
		for (; t$2 < e$2.length;) r$2 = 101 * r$2 + e$2.charCodeAt(t$2++) >>> 0;
		return "go" + r$2;
	})(u$1));
	if (!c[d$1]) {
		let t$2 = u$1 !== e$1 ? e$1 : ((e$2) => {
			let t$3, r$2, o$1 = [{}];
			for (; t$3 = l.exec(e$2.replace(a, ""));) t$3[4] ? o$1.shift() : t$3[3] ? (r$2 = t$3[3].replace(n$1, " ").trim(), o$1.unshift(o$1[0][r$2] = o$1[0][r$2] || {})) : o$1[0][t$3[1]] = t$3[2].replace(n$1, " ").trim();
			return o$1[0];
		})(e$1);
		c[d$1] = o(p$1 ? { ["@keyframes " + d$1]: t$2 } : t$2, r$1 ? "" : "." + d$1);
	}
	let f$2 = r$1 && c.g ? c.g : null;
	return r$1 && (c.g = c[d$1]), ((e$2, t$2, r$2, l$1) => {
		l$1 ? t$2.data = t$2.data.replace(l$1, e$2) : -1 === t$2.data.indexOf(e$2) && (t$2.data = r$2 ? e$2 + t$2.data : t$2.data + e$2);
	})(c[d$1], t$1, i$1, f$2), d$1;
}, p = (e$1, t$1, r$1) => e$1.reduce((e$2, l$1, a$1) => {
	let n$2 = t$1[a$1];
	if (n$2 && n$2.call) {
		let e$3 = n$2(r$1), t$2 = e$3 && e$3.props && e$3.props.className || /^go/.test(e$3) && e$3;
		n$2 = t$2 ? "." + t$2 : e$3 && "object" == typeof e$3 ? e$3.props ? "" : o(e$3, "") : !1 === e$3 ? "" : e$3;
	}
	return e$2 + l$1 + (null == n$2 ? "" : n$2);
}, "");
function u(e$1) {
	let r$1 = this || {}, l$1 = e$1.call ? e$1(r$1.p) : e$1;
	return i(l$1.unshift ? l$1.raw ? p(l$1, [].slice.call(arguments, 1), r$1.p) : l$1.reduce((e$2, t$1) => Object.assign(e$2, t$1 && t$1.call ? t$1(r$1.p) : t$1), {}) : l$1, t(r$1.target), r$1.g, r$1.o, r$1.k);
}
var d, f$1, g, b = u.bind({ g: 1 }), h$1 = u.bind({ k: 1 });
function m(e$1, t$1, r$1, l$1) {
	o.p = t$1, d = e$1, f$1 = r$1, g = l$1;
}
function w$1(e$1, t$1) {
	let r$1 = this || {};
	return function() {
		let l$1 = arguments;
		function a$1(n$2, o$1) {
			let c$1 = Object.assign({}, n$2), s$1 = c$1.className || a$1.className;
			r$1.p = Object.assign({ theme: f$1 && f$1() }, c$1), r$1.o = / *go\d+/.test(s$1), c$1.className = u.apply(r$1, l$1) + (s$1 ? " " + s$1 : ""), t$1 && (c$1.ref = o$1);
			let i$1 = e$1;
			return e$1[0] && (i$1 = c$1.as || e$1, delete c$1.as), g && i$1[0] && g(c$1), d(i$1, c$1);
		}
		return t$1 ? t$1(a$1) : a$1;
	};
}

//#endregion
//#region node_modules/react-hot-toast/dist/index.mjs
var Z = (e$1) => typeof e$1 == "function", h = (e$1, t$1) => Z(e$1) ? e$1(t$1) : e$1;
var W = (() => {
	let e$1 = 0;
	return () => (++e$1).toString();
})(), E = (() => {
	let e$1;
	return () => {
		if (e$1 === void 0 && typeof window < "u") {
			let t$1 = matchMedia("(prefers-reduced-motion: reduce)");
			e$1 = !t$1 || t$1.matches;
		}
		return e$1;
	};
})();
var re = 20, k = "default";
var H = (e$1, t$1) => {
	let { toastLimit: o$1 } = e$1.settings;
	switch (t$1.type) {
		case 0: return {
			...e$1,
			toasts: [t$1.toast, ...e$1.toasts].slice(0, o$1)
		};
		case 1: return {
			...e$1,
			toasts: e$1.toasts.map((r$1) => r$1.id === t$1.toast.id ? {
				...r$1,
				...t$1.toast
			} : r$1)
		};
		case 2:
			let { toast: s$1 } = t$1;
			return H(e$1, {
				type: e$1.toasts.find((r$1) => r$1.id === s$1.id) ? 1 : 0,
				toast: s$1
			});
		case 3:
			let { toastId: a$1 } = t$1;
			return {
				...e$1,
				toasts: e$1.toasts.map((r$1) => r$1.id === a$1 || a$1 === void 0 ? {
					...r$1,
					dismissed: !0,
					visible: !1
				} : r$1)
			};
		case 4: return t$1.toastId === void 0 ? {
			...e$1,
			toasts: []
		} : {
			...e$1,
			toasts: e$1.toasts.filter((r$1) => r$1.id !== t$1.toastId)
		};
		case 5: return {
			...e$1,
			pausedAt: t$1.time
		};
		case 6:
			let i$1 = t$1.time - (e$1.pausedAt || 0);
			return {
				...e$1,
				pausedAt: void 0,
				toasts: e$1.toasts.map((r$1) => ({
					...r$1,
					pauseDuration: r$1.pauseDuration + i$1
				}))
			};
	}
}, v = [], j = {
	toasts: [],
	pausedAt: void 0,
	settings: { toastLimit: re }
}, f = {}, Y = (e$1, t$1 = k) => {
	f[t$1] = H(f[t$1] || j, e$1), v.forEach(([o$1, s$1]) => {
		o$1 === t$1 && s$1(f[t$1]);
	});
}, _ = (e$1) => Object.keys(f).forEach((t$1) => Y(e$1, t$1)), Q = (e$1) => Object.keys(f).find((t$1) => f[t$1].toasts.some((o$1) => o$1.id === e$1)), S = (e$1 = k) => (t$1) => {
	Y(t$1, e$1);
}, se = {
	blank: 4e3,
	error: 4e3,
	success: 2e3,
	loading: Infinity,
	custom: 4e3
}, V = (e$1 = {}, t$1 = k) => {
	let [o$1, s$1] = (0, import_react.useState)(f[t$1] || j), a$1 = (0, import_react.useRef)(f[t$1]);
	(0, import_react.useEffect)(() => (a$1.current !== f[t$1] && s$1(f[t$1]), v.push([t$1, s$1]), () => {
		let r$1 = v.findIndex(([l$1]) => l$1 === t$1);
		r$1 > -1 && v.splice(r$1, 1);
	}), [t$1]);
	let i$1 = o$1.toasts.map((r$1) => {
		var l$1, g$1, T;
		return {
			...e$1,
			...e$1[r$1.type],
			...r$1,
			removeDelay: r$1.removeDelay || ((l$1 = e$1[r$1.type]) == null ? void 0 : l$1.removeDelay) || (e$1 == null ? void 0 : e$1.removeDelay),
			duration: r$1.duration || ((g$1 = e$1[r$1.type]) == null ? void 0 : g$1.duration) || (e$1 == null ? void 0 : e$1.duration) || se[r$1.type],
			style: {
				...e$1.style,
				...(T = e$1[r$1.type]) == null ? void 0 : T.style,
				...r$1.style
			}
		};
	});
	return {
		...o$1,
		toasts: i$1
	};
};
var ie = (e$1, t$1 = "blank", o$1) => ({
	createdAt: Date.now(),
	visible: !0,
	dismissed: !1,
	type: t$1,
	ariaProps: {
		role: "status",
		"aria-live": "polite"
	},
	message: e$1,
	pauseDuration: 0,
	...o$1,
	id: (o$1 == null ? void 0 : o$1.id) || W()
}), P = (e$1) => (t$1, o$1) => {
	let s$1 = ie(t$1, e$1, o$1);
	return S(s$1.toasterId || Q(s$1.id))({
		type: 2,
		toast: s$1
	}), s$1.id;
}, n = (e$1, t$1) => P("blank")(e$1, t$1);
n.error = P("error");
n.success = P("success");
n.loading = P("loading");
n.custom = P("custom");
n.dismiss = (e$1, t$1) => {
	let o$1 = {
		type: 3,
		toastId: e$1
	};
	t$1 ? S(t$1)(o$1) : _(o$1);
};
n.dismissAll = (e$1) => n.dismiss(void 0, e$1);
n.remove = (e$1, t$1) => {
	let o$1 = {
		type: 4,
		toastId: e$1
	};
	t$1 ? S(t$1)(o$1) : _(o$1);
};
n.removeAll = (e$1) => n.remove(void 0, e$1);
n.promise = (e$1, t$1, o$1) => {
	let s$1 = n.loading(t$1.loading, {
		...o$1,
		...o$1 == null ? void 0 : o$1.loading
	});
	return typeof e$1 == "function" && (e$1 = e$1()), e$1.then((a$1) => {
		let i$1 = t$1.success ? h(t$1.success, a$1) : void 0;
		return i$1 ? n.success(i$1, {
			id: s$1,
			...o$1,
			...o$1 == null ? void 0 : o$1.success
		}) : n.dismiss(s$1), a$1;
	}).catch((a$1) => {
		let i$1 = t$1.error ? h(t$1.error, a$1) : void 0;
		i$1 ? n.error(i$1, {
			id: s$1,
			...o$1,
			...o$1 == null ? void 0 : o$1.error
		}) : n.dismiss(s$1);
	}), e$1;
};
var ce = 1e3, w = (e$1, t$1 = "default") => {
	let { toasts: o$1, pausedAt: s$1 } = V(e$1, t$1), a$1 = (0, import_react.useRef)(/* @__PURE__ */ new Map()).current, i$1 = (0, import_react.useCallback)((c$1, m$1 = ce) => {
		if (a$1.has(c$1)) return;
		let p$1 = setTimeout(() => {
			a$1.delete(c$1), r$1({
				type: 4,
				toastId: c$1
			});
		}, m$1);
		a$1.set(c$1, p$1);
	}, []);
	(0, import_react.useEffect)(() => {
		if (s$1) return;
		let c$1 = Date.now(), m$1 = o$1.map((p$1) => {
			if (p$1.duration === Infinity) return;
			let R = (p$1.duration || 0) + p$1.pauseDuration - (c$1 - p$1.createdAt);
			if (R < 0) {
				p$1.visible && n.dismiss(p$1.id);
				return;
			}
			return setTimeout(() => n.dismiss(p$1.id, t$1), R);
		});
		return () => {
			m$1.forEach((p$1) => p$1 && clearTimeout(p$1));
		};
	}, [
		o$1,
		s$1,
		t$1
	]);
	let r$1 = (0, import_react.useCallback)(S(t$1), [t$1]), l$1 = (0, import_react.useCallback)(() => {
		r$1({
			type: 5,
			time: Date.now()
		});
	}, [r$1]), g$1 = (0, import_react.useCallback)((c$1, m$1) => {
		r$1({
			type: 1,
			toast: {
				id: c$1,
				height: m$1
			}
		});
	}, [r$1]), T = (0, import_react.useCallback)(() => {
		s$1 && r$1({
			type: 6,
			time: Date.now()
		});
	}, [s$1, r$1]), d$1 = (0, import_react.useCallback)((c$1, m$1) => {
		let { reverseOrder: p$1 = !1, gutter: R = 8, defaultPosition: z } = m$1 || {}, O = o$1.filter((u$1) => (u$1.position || z) === (c$1.position || z) && u$1.height), K = O.findIndex((u$1) => u$1.id === c$1.id), B = O.filter((u$1, I) => I < K && u$1.visible).length;
		return O.filter((u$1) => u$1.visible).slice(...p$1 ? [B + 1] : [0, B]).reduce((u$1, I) => u$1 + (I.height || 0) + R, 0);
	}, [o$1]);
	return (0, import_react.useEffect)(() => {
		o$1.forEach((c$1) => {
			if (c$1.dismissed) i$1(c$1.id, c$1.removeDelay);
			else {
				let m$1 = a$1.get(c$1.id);
				m$1 && (clearTimeout(m$1), a$1.delete(c$1.id));
			}
		});
	}, [o$1, i$1]), {
		toasts: o$1,
		handlers: {
			updateHeight: g$1,
			startPause: l$1,
			endPause: T,
			calculateOffset: d$1
		}
	};
};
var de = h$1`
from {
  transform: scale(0) rotate(45deg);
	opacity: 0;
}
to {
 transform: scale(1) rotate(45deg);
  opacity: 1;
}`, me = h$1`
from {
  transform: scale(0);
  opacity: 0;
}
to {
  transform: scale(1);
  opacity: 1;
}`, le = h$1`
from {
  transform: scale(0) rotate(90deg);
	opacity: 0;
}
to {
  transform: scale(1) rotate(90deg);
	opacity: 1;
}`, C = w$1("div")`
  width: 20px;
  opacity: 0;
  height: 20px;
  border-radius: 10px;
  background: ${(e$1) => e$1.primary || "#ff4b4b"};
  position: relative;
  transform: rotate(45deg);

  animation: ${de} 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)
    forwards;
  animation-delay: 100ms;

  &:after,
  &:before {
    content: '';
    animation: ${me} 0.15s ease-out forwards;
    animation-delay: 150ms;
    position: absolute;
    border-radius: 3px;
    opacity: 0;
    background: ${(e$1) => e$1.secondary || "#fff"};
    bottom: 9px;
    left: 4px;
    height: 2px;
    width: 12px;
  }

  &:before {
    animation: ${le} 0.15s ease-out forwards;
    animation-delay: 180ms;
    transform: rotate(90deg);
  }
`;
var Te = h$1`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`, F = w$1("div")`
  width: 12px;
  height: 12px;
  box-sizing: border-box;
  border: 2px solid;
  border-radius: 100%;
  border-color: ${(e$1) => e$1.secondary || "#e0e0e0"};
  border-right-color: ${(e$1) => e$1.primary || "#616161"};
  animation: ${Te} 1s linear infinite;
`;
var ge = h$1`
from {
  transform: scale(0) rotate(45deg);
	opacity: 0;
}
to {
  transform: scale(1) rotate(45deg);
	opacity: 1;
}`, he = h$1`
0% {
	height: 0;
	width: 0;
	opacity: 0;
}
40% {
  height: 0;
	width: 6px;
	opacity: 1;
}
100% {
  opacity: 1;
  height: 10px;
}`, L = w$1("div")`
  width: 20px;
  opacity: 0;
  height: 20px;
  border-radius: 10px;
  background: ${(e$1) => e$1.primary || "#61d345"};
  position: relative;
  transform: rotate(45deg);

  animation: ${ge} 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)
    forwards;
  animation-delay: 100ms;
  &:after {
    content: '';
    box-sizing: border-box;
    animation: ${he} 0.2s ease-out forwards;
    opacity: 0;
    animation-delay: 200ms;
    position: absolute;
    border-right: 2px solid;
    border-bottom: 2px solid;
    border-color: ${(e$1) => e$1.secondary || "#fff"};
    bottom: 6px;
    left: 6px;
    height: 10px;
    width: 6px;
  }
`;
var be = w$1("div")`
  position: absolute;
`, Se = w$1("div")`
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  min-width: 20px;
  min-height: 20px;
`, Ae = h$1`
from {
  transform: scale(0.6);
  opacity: 0.4;
}
to {
  transform: scale(1);
  opacity: 1;
}`, Pe = w$1("div")`
  position: relative;
  transform: scale(0.6);
  opacity: 0.4;
  min-width: 20px;
  animation: ${Ae} 0.3s 0.12s cubic-bezier(0.175, 0.885, 0.32, 1.275)
    forwards;
`, $ = ({ toast: e$1 }) => {
	let { icon: t$1, type: o$1, iconTheme: s$1 } = e$1;
	return t$1 !== void 0 ? typeof t$1 == "string" ? import_react.createElement(Pe, null, t$1) : t$1 : o$1 === "blank" ? null : import_react.createElement(Se, null, import_react.createElement(F, { ...s$1 }), o$1 !== "loading" && import_react.createElement(be, null, o$1 === "error" ? import_react.createElement(C, { ...s$1 }) : import_react.createElement(L, { ...s$1 })));
};
var Re = (e$1) => `
0% {transform: translate3d(0,${e$1 * -200}%,0) scale(.6); opacity:.5;}
100% {transform: translate3d(0,0,0) scale(1); opacity:1;}
`, Ee = (e$1) => `
0% {transform: translate3d(0,0,-1px) scale(1); opacity:1;}
100% {transform: translate3d(0,${e$1 * -150}%,-1px) scale(.6); opacity:0;}
`, ve = "0%{opacity:0;} 100%{opacity:1;}", De = "0%{opacity:1;} 100%{opacity:0;}", Oe = w$1("div")`
  display: flex;
  align-items: center;
  background: #fff;
  color: #363636;
  line-height: 1.3;
  will-change: transform;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1), 0 3px 3px rgba(0, 0, 0, 0.05);
  max-width: 350px;
  pointer-events: auto;
  padding: 8px 10px;
  border-radius: 8px;
`, Ie = w$1("div")`
  display: flex;
  justify-content: center;
  margin: 4px 10px;
  color: inherit;
  flex: 1 1 auto;
  white-space: pre-line;
`, ke = (e$1, t$1) => {
	let s$1 = e$1.includes("top") ? 1 : -1, [a$1, i$1] = E() ? [ve, De] : [Re(s$1), Ee(s$1)];
	return { animation: t$1 ? `${h$1(a$1)} 0.35s cubic-bezier(.21,1.02,.73,1) forwards` : `${h$1(i$1)} 0.4s forwards cubic-bezier(.06,.71,.55,1)` };
}, N = import_react.memo(({ toast: e$1, position: t$1, style: o$1, children: s$1 }) => {
	let a$1 = e$1.height ? ke(e$1.position || t$1 || "top-center", e$1.visible) : { opacity: 0 }, i$1 = import_react.createElement($, { toast: e$1 }), r$1 = import_react.createElement(Ie, { ...e$1.ariaProps }, h(e$1.message, e$1));
	return import_react.createElement(Oe, {
		className: e$1.className,
		style: {
			...a$1,
			...o$1,
			...e$1.style
		}
	}, typeof s$1 == "function" ? s$1({
		icon: i$1,
		message: r$1
	}) : import_react.createElement(import_react.Fragment, null, i$1, r$1));
});
m(import_react.createElement);
var we = ({ id: e$1, className: t$1, style: o$1, onHeightUpdate: s$1, children: a$1 }) => {
	let i$1 = import_react.useCallback((r$1) => {
		if (r$1) {
			let l$1 = () => {
				let g$1 = r$1.getBoundingClientRect().height;
				s$1(e$1, g$1);
			};
			l$1(), new MutationObserver(l$1).observe(r$1, {
				subtree: !0,
				childList: !0,
				characterData: !0
			});
		}
	}, [e$1, s$1]);
	return import_react.createElement("div", {
		ref: i$1,
		className: t$1,
		style: o$1
	}, a$1);
}, Me = (e$1, t$1) => {
	let o$1 = e$1.includes("top"), s$1 = o$1 ? { top: 0 } : { bottom: 0 }, a$1 = e$1.includes("center") ? { justifyContent: "center" } : e$1.includes("right") ? { justifyContent: "flex-end" } : {};
	return {
		left: 0,
		right: 0,
		display: "flex",
		position: "absolute",
		transition: E() ? void 0 : "all 230ms cubic-bezier(.21,1.02,.73,1)",
		transform: `translateY(${t$1 * (o$1 ? 1 : -1)}px)`,
		...s$1,
		...a$1
	};
}, Ce = u`
  z-index: 9999;
  > * {
    pointer-events: auto;
  }
`, D = 16, Fe = ({ reverseOrder: e$1, position: t$1 = "top-center", toastOptions: o$1, gutter: s$1, children: a$1, toasterId: i$1, containerStyle: r$1, containerClassName: l$1 }) => {
	let { toasts: g$1, handlers: T } = w(o$1, i$1);
	return import_react.createElement("div", {
		"data-rht-toaster": i$1 || "",
		style: {
			position: "fixed",
			zIndex: 9999,
			top: D,
			left: D,
			right: D,
			bottom: D,
			pointerEvents: "none",
			...r$1
		},
		className: l$1,
		onMouseEnter: T.startPause,
		onMouseLeave: T.endPause
	}, g$1.map((d$1) => {
		let c$1 = d$1.position || t$1, m$1 = T.calculateOffset(d$1, {
			reverseOrder: e$1,
			gutter: s$1,
			defaultPosition: t$1
		}), p$1 = Me(c$1, m$1);
		return import_react.createElement(we, {
			id: d$1.id,
			key: d$1.id,
			onHeightUpdate: T.updateHeight,
			className: d$1.visible ? Ce : "",
			style: p$1
		}, d$1.type === "custom" ? h(d$1.message, d$1) : a$1 ? a$1(d$1) : import_react.createElement(N, {
			toast: d$1,
			position: c$1
		}));
	}));
};
var zt = n;

//#endregion
export { L as CheckmarkIcon, C as ErrorIcon, F as LoaderIcon, N as ToastBar, $ as ToastIcon, Fe as Toaster, zt as default, h as resolveValue, n as toast, w as useToaster, V as useToasterStore };
//# sourceMappingURL=react-hot-toast.js.map