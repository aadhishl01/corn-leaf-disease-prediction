import mongoose from 'mongoose';
import userModel from './models/usermodel.js';
import { config } from 'dotenv';
config({ path: './.env' });

const listUsers = async () => {
    try {
        await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/ecommerce');
        console.log('Connected to DB');
        const users = await userModel.find({}, { email: 1, _id: 0 });
        console.log('Registered emails:');
        users.forEach(user => console.log(user.email));
        process.exit(0);
    } catch (error) {
        console.log('Error:', error.message);
        process.exit(1);
    }
};

listUsers();
