const express = require('express');
const router = express.Router();


router.use(express.urlencoded({ extended: true }));
router.use(express.json());
let Animal = require('../models/animal');

router.get('/', async (req, res) => {

    my_animals = []
    try {
        my_animals = await Animal.find();

        console.log(my_animals, "my_animals")

    }
    catch (error) {
        console.log("error", error)

    }
    res.render('farm', { cats: my_animals });

});







module.exports = router; //always forget