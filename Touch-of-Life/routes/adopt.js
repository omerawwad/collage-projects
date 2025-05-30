const express = require('express');
const router = express.Router();


router.use(express.urlencoded({ extended: true }));
router.use(express.json());
let Animal = require('../models/animal');

router.get('/', async (req, res) => {

res.render('adopt')

});







module.exports = router; //always forget