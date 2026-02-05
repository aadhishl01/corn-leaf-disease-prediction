"use strict";
// File generated from our OpenAPI spec
Object.defineProperty(exports, "__esModule", { value: true });
exports.PaymentAttemptRecords = void 0;
const StripeResource_js_1 = require("../StripeResource.js");
const stripeMethod = StripeResource_js_1.StripeResource.method;
exports.PaymentAttemptRecords = StripeResource_js_1.StripeResource.extend({
    retrieve: stripeMethod({
        method: 'GET',
        fullPath: '/v1/payment_attempt_records/{id}',
    }),
    list: stripeMethod({
        method: 'GET',
        fullPath: '/v1/payment_attempt_records',
        methodType: 'list',
    }),
});
