import React, { createContext, useState, useEffect } from "react";
import { products } from "../assets/assets";
import toast from 'react-hot-toast';
import { useNavigate } from "react-router-dom";
import { auth, db } from "../config/firebase";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { doc, getDoc, setDoc, updateDoc, onSnapshot } from "firebase/firestore";

export const ShopContext = createContext();

const ShopContextProvider = (props) => {

    const currency = '₹';
    const delivery_fee = 10;
    const [search, setsearch] = useState('');
    const [showsearch, setshowsearch] = useState(false);
    const [cartItems, setCartItems] = useState({});
    const [user, setUser] = useState(null);
    const navigate = useNavigate();

    // Listen for auth state changes
    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
            setUser(currentUser);
            if (currentUser) {
                // Load cart from Firestore
                const userDocRef = doc(db, "users", currentUser.uid);
                const userDoc = await getDoc(userDocRef);
                if (userDoc.exists() && userDoc.data().cart) {
                    setCartItems(userDoc.data().cart);
                } else {
                    // Create user doc if not exists
                    if (!userDoc.exists()) {
                        await setDoc(userDocRef, { cart: {}, email: currentUser.email });
                    }
                    setCartItems({});
                }
            } else {
                setCartItems({});
            }
        });
        return () => unsubscribe();
    }, []);

    // Sync cart to Firestore whenever it changes (if user is logged in)
    useEffect(() => {
        if (user) {
            const userDocRef = doc(db, "users", user.uid);
            updateDoc(userDocRef, { cart: cartItems }).catch(err => console.error("Error updating cart:", err));
        }
    }, [cartItems, user]);

    const addToCart = async (itemId, size) => {
        if (!size) {
            toast.error('Select Product Size');
            return;
        }
        let cartData = JSON.parse(JSON.stringify(cartItems));

        if (cartData[itemId]) {
            if (cartData[itemId][size]) {
                cartData[itemId][size] += 1;
            } else {
                cartData[itemId][size] = 1;
            }
        } else {
            cartData[itemId] = {};
            cartData[itemId][size] = 1;
        }
        setCartItems(cartData);
        toast.success('Added to cart');
    };

    const getcartcount = () => {
        let totalcount = 0;
        for (const items in cartItems) {
            for (const item in cartItems[items]) {
                try {
                    if (cartItems[items][item] > 0) {
                        totalcount += cartItems[items][item];
                    }
                } catch (error) {
                    // ignore
                }
            }
        }
        return totalcount;
    };

    const updateQuantity = async (itemId, size, quantity) => {
        let cartData = structuredClone(cartItems);
        cartData[itemId][size] = quantity;
        setCartItems(cartData);
    };

    const getCartCount = () => {
        let totalCount = 0;
        for (const items in cartItems) {
            for (const item in cartItems[items]) {
                try {
                    if (cartItems[items][item] > 0) {
                        totalCount += cartItems[items][item];
                    }
                } catch (error) {
                    // ignore
                }
            }
        }
        return totalCount;
    };

    const getCartAmount = () => {
        let totalAmount = 0;
        for (const items in cartItems) {
            let itemInfo = products.find((product) => product._id === items);
            if (itemInfo) {
                for (const item in cartItems[items]) {
                    try {
                        if (cartItems[items][item] > 0) {
                            totalAmount += itemInfo.price * cartItems[items][item];
                        }
                    } catch (error) {
                        // ignore
                    }
                }
            }
        }
        return totalAmount;
    };

    const logout = async () => {
        try {
            await signOut(auth);
            toast.success("Logged out successfully");
            navigate('/login');
        } catch (error) {
            toast.error(error.message);
        }
    };

    const value = {
        products: products || [],
        currency,
        delivery_fee,
        search,
        setsearch,
        showsearch,
        setshowsearch,
        cartItems,
        setCartItems,
        addToCart,
        getcartcount,
        getCartCount,
        updateQuantity,
        getCartAmount,
        navigate,
        user,
        logout
    };

    return (
        <ShopContext.Provider value={value}>
            {props.children}
        </ShopContext.Provider>
    );
};

export default ShopContextProvider;
