if (process.env.NODE_ENV !== 'production') {
    require('dotenv').config()
}
const express = require("express");
const app = express();
const mongoose = require('mongoose');
const passport = require('passport');
const session = require('express-session')
const uri = "mongodb+srv://TOF:TOF1761@cluster0.mzkc4h3.mongodb.net/?retryWrites=true&w=majority"
async function connect() {
    try {
        await mongoose.connect(uri);
        console.log('DB Connected successfully')
    }
    catch (err) {
        console.error(err);
    }
}

connect();
require('./passport-config.js')(passport)

app.set('view engine', 'ejs');
app.use(express.json());//middleware to convert req.body m to json


app.use(express.static(__dirname + '/public'));
app.use(express.urlencoded({ extended: false }));
app.use(session({
    secret: 'asdh',
    resave: false,
    saveUninitialized: false
}));

var bodyParser = require('body-parser')

// parse application/x-www-form-urlencoded
app.use(bodyParser.urlencoded({ extended: false }))

// parse application/json
app.use(bodyParser.json())

app.use(passport.initialize());
app.use(passport.session());



let request = require('./models/requests');


app.get('/', (req, res) => {
    res.render('index')
});



app.get('/login', (req, res) => {
    res.render('login')
});

app.post('/login', passport.authenticate('local', {
    successRedirect: '/admin',
    failureRedirect: '/login',

}))

app.get('/admin', checkAuth, (req, res) => {
    res.render('admin')
})


//admin post
app.post('/admin', checkAuth, async (req, res) => {
    
    console.log(req.body,"req query");
    let result;
    if(req.body.addordel === "add")
    {
        let newAnimal = new Animal({
            name: req.body.animalName,
            specie: req.body.animalSpecie,
            breed:req.body.animalBreed,
            imgUrl: req.body.imgurl,
        })
        try{
        result = await newAnimal.save();
    }
    catch(err){
        console.log(err)
    }
    }
    else
    {
        let ID = req.body.animalid;
        console.log(ID,"sent ID");
        try{
        result = await Animal.deleteOne({_id:ID});
    }
    catch(err)
    {
        console.log(err)
    }
    }
    console.log (result);

    
    res.render('admin')
})

app.get('/areqs', checkAuth, async (req, res) => {
    
    try {
        let result = await request.find();
        console.log(result , "result of requests");
        res.render('areqs', { rqs: result });

    }
    catch (err) {
        console.log(err)
        res.render('areqs', { rqs: [] })
    }
})


app.post('/delete',checkAuth ,async (req, res) => {
    console.log(req.body)
    let deleteResult =await request.findByIdAndDelete(req.body.reqid)
    console.log(deleteResult)
    try {
        let result = await request.find();
        res.render('areqs', { rqs: result });
        }
    catch (err) {
        console.log(err)
        res.render('areqs', { rqs: [] })
        }
    })



function checkAuth(req, res, next) {

    if (req.isAuthenticated()) {
        return next()
    }

    res.redirect('/login')

}

app.post('/logout', function (req, res, next) {
    req.logout(function (err) {
        if (err) { return next(err); }
        res.redirect('/');
    });
});




app.get('/payment', (req, res) => {
    //res.send(__dirname+'/payment.html')
    res.sendFile(__dirname+'/payment.html')
})
app.get('/payment200', (req, res) => {
    //res.send(__dirname+'/payment.html')
    res.sendFile(__dirname+'/payment200.html')
})
app.get('/payment500', (req, res) => {
    //res.send(__dirname+'/payment.html')
    res.sendFile(__dirname+'/payment500.html')
})

app.post("/adoptrequest",async (req,res) => {
console.log(req.body , "adopt request body")
    
try{
let newreq =  new request({
        name : req.body.name,
        phone : req.body.phone_number,
        animalid : req.body.animalid

    });
    let result = await newreq.save()
    console.log(result)
    res.render("adopt")
}
catch(err){
    res.send("error")
}
})




const adopt = require('./routes/adopt');
const farm = require('./routes/farm');
const Animal = require('./models/animal.js');
app.use('/adopt', adopt);
app.use('/farm', farm);
const port = process.env.PORT || 3000;
app.listen(port, console.log("LISTENING ON PORT : ", port));
