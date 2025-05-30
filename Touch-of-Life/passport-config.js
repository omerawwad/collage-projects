

const User = require('./models/account')

const LocalStrategy = require('passport-local').Strategy;
module.exports = passportConfig = (passport)=> {
    passport.use(new LocalStrategy({ usernameField : 'email'} , (email,password,done) => {

        
        const user = User.find({email :email} , (error,data)=>{
            if(error) {
                console.error(error);
            }
            else {
                if(data.length ==0) 
                {
                    console.log("not data")
                    return done(null,false,{message : "user not found"})
                }
                
                if(error) console.error(error);
                console.log(data);
                console.log(data[0].email,"mail f");
                console.log(data[0].password,"pass f");
                console.log(password,"pass");
                console.log(email,"mail");
                if(data[0].password !== password)
                return done(null,false,{message : "user not found"})

     //
                console.log("good")
                return done(null,data)
                               
            }   
        }
        );
        


    }));

    passport.serializeUser((user,done)=> done(null,user));
    passport.deserializeUser((user,done)=> done(null,user));
}
