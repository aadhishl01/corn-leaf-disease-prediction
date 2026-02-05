import { v2 as cloudinary } from 'cloudinary';

const connectCloudinary = async () => {
    cloudinary.config({
        cloud_name: process.env.CLOUDINARY_NAME,
        api_key: process.env.CLOUDINARY_API_KEY,
        api_secret: process.env.CLOUDINARY_SECRET_KEY,
    });

    // Test connection
    try {
        const result = await cloudinary.api.ping();
        console.log('Cloudinary connected successfully:', result);
    } catch (error) {
        console.log('Cloudinary connection failed:', error.message);
    }
};

export default connectCloudinary;
