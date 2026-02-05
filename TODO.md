# TODO: Fix SSL Error in Ecommerce App Backend

## Steps to Complete:
- [x] Update `backend/generate_cert.js` to generate a proper self-signed X.509 certificate instead of just a public key.
- [x] Modify `backend/server.js` to create an HTTPS server using the generated certificates.
- [x] Run the certificate generation script to create new cert.pem and key.pem.
- [x] Start the HTTPS server and verify it runs without SSL errors.
