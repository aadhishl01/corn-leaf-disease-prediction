"use strict";
// File generated from our OpenAPI spec
Object.defineProperty(exports, "__esModule", { value: true });
exports.BalanceSettings = void 0;
const StripeResource_js_1 = require("../StripeResource.js");
const stripeMethod = StripeResource_js_1.StripeResource.method;
exports.BalanceSettings = StripeResource_js_1.StripeResource.extend({
    retrieve: stripeMethod({ method: 'GET', fullPath: '/v1/balance_settings' }),
    update: stripeMethod({ method: 'POST', fullPath: '/v1/balance_settings' }),
});
