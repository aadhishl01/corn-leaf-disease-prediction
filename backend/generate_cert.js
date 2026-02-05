import selfsigned from 'selfsigned';
import fs from 'fs';

const attrs = [
  { name: 'commonName', value: 'localhost' },
  { name: 'countryName', value: 'US' },
  { name: 'stateOrProvinceName', value: 'State' },
  { name: 'localityName', value: 'City' },
  { name: 'organizationName', value: 'Organization' }
];

const pems = selfsigned.generate(attrs, { days: 365 });

fs.writeFileSync('key.pem', pems.private);
fs.writeFileSync('cert.pem', pems.cert);

console.log('Self-signed certificates generated');
