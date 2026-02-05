import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore, enableIndexedDbPersistence } from "firebase/firestore";

const firebaseConfig = {
    apiKey: "AIzaSyCA7hF-icXPBOoWfP07U8Xn6IhHUF30SwI",
    authDomain: "ecommerce-6c4ac.firebaseapp.com",
    projectId: "ecommerce-6c4ac",
    storageBucket: "ecommerce-6c4ac.firebasestorage.app",
    messagingSenderId: "599922161766",
    appId: "1:599922161766:web:58e551c78d55d0d69055a7",
    measurementId: "G-Z4SVS7XX9X"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const googleProvider = new GoogleAuthProvider();

// Enable offline persistence
enableIndexedDbPersistence(db)
    .catch((err) => {
        if (err.code == 'failed-precondition') {
            console.warn('Persistence failed: Multiple tabs open');
        } else if (err.code == 'unimplemented') {
            console.warn('Persistence not supported by browser');
        }
    });

export { auth, db, googleProvider };
