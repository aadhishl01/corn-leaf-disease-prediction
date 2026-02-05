import express from 'express'
import cors from 'cors'
import { config } from 'dotenv'
config()
import http from 'http'
import connectDB from './config/mongodb.js'
import connectCloudinary from './config/cloudinary.js'
import userRouter from './routes/userRoute.js'
import productRouter from './routes/productRoute.js'


//app config
const app=express()
const port= 4000
await connectDB()
connectCloudinary()

// middleware
app.use(express.json())
app.use(cors())

//api endpoints
app.use('/api/user',userRouter)
app.use('/api/product',productRouter)

app.get('/',(req,res)=>{
    res.send(" API WORKING ")
})

// Create HTTP server
const server = http.createServer(app);

// Start HTTP server on port 4000
server.listen(port, '0.0.0.0', () => {
    console.log('HTTP SERVER STARTED ON PORT: ' + port);
});
