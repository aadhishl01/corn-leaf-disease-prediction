import mongoose from "mongoose";
import { MongoMemoryServer } from 'mongodb-memory-server';

const connectDB = async () => {
    mongoose.connection.on('connected', () => {
        console.log('DB connected');
    });

    const mongoURI = process.env.MONGODB_URI || 'mongodb://localhost:27017';
    try {
        await mongoose.connect(mongoURI, {
            useNewUrlParser: true,
            useUnifiedTopology: true,
            serverSelectionTimeoutMS: 5000,
            socketTimeoutMS: 45000,
            bufferCommands: true  // Changed to true to allow buffering
        });
    } catch (error) {
        console.log('MongoDB connection failed:', error.message);
        console.log('Server will continue without database connection.');
    }
};

export default connectDB;
