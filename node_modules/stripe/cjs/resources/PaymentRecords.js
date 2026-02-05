"use strict";
// File generated from our OpenAPI spec
Object.defineProperty(exports, "__esModule", { value: true });
exports.PaymentRecords = void 0;
const StripeResource_js_1 = require("../StripeResource.js");
const stripeMethod = StripeResource_js_1.StripeResource.method;
exports.PaymentRecords = StripeResource_js_1.StripeResource.extend({
    retrieve: stripeMethod({ method: 'GET', fullPath: '/v1/payment_records/{id}' }),
    reportPayment: stripeMethod({
        method: 'POST',
        fullPath: '/v1/payment_records/report_payment',
    }),
    reportPaymentAttempt: stripeMethod({
        method: 'POST',
        fullPath: '/v1/payment_records/{id}/report_payment_attempt',
    }),
    reportPaymentAttemptCanceled: stripeMethod({
        method: 'POST',
        fullPath: '/v1/payment_records/{id}/report_payment_attempt_canceled',
    }),
    reportPaymentAttemptFailed: stripeMethod({
        method: 'POST',
        fullPath: '/v1/payment_records/{id}/report_payment_attempt_failed',
    }),
    reportPaymentAttemptGuaranteed: stripeMethod({
        method: 'POST',
        fullPath: '/v1/payment_records/{id}/report_payment_attempt_guaranteed',
    }),
    reportPaymentAttemptInformational: stripeMethod({
        method: 'POST',
        fullPath: '/v1/payment_records/{id}/report_payment_attempt_informational',
    }),
    reportRefund: stripeMethod({
        method: 'POST',
        fullPath: '/v1/payment_records/{id}/report_refund',
    }),
});
