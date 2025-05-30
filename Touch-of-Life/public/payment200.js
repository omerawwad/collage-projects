ammount = 200
const apiKey = "ZXlKaGJHY2lPaUpJVXpVeE1pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SndjbTltYVd4bFgzQnJJam8yTlRNM01UZ3NJbU5zWVhOeklqb2lUV1Z5WTJoaGJuUWlMQ0p1WVcxbElqb2lhVzVwZEdsaGJDSjkuNjRLZXRvTE0wU3MyQTZVUVdFWm94eTRtcm5vZlUxQ1VMc1d3ZERsYUdnQmFCdGhEaTE2a3ljdU9Pb0FQU3FUU0tXejVvbXlQSkp6R2xVX3VXcE1yUXc="

async function firstStep() {
    let data = {
        "api_key": apiKey
    }

    let request = fetch("https://accept.paymob.com/api/auth/tokens", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })

    let response = await request;
    let finalres = await response.json();
    token = finalres.token;
    console.log("token", token)
    secondStep(token)
}




async function secondStep(token) {
    let data = {
        "auth_token": token,
        "delivery_needed": "false",
        "amount_cents": `${ammount*100}`,
        "currency": "EGP",
        "items": [],

    }
    let request = fetch("https://accept.paymob.com/api/ecommerce/orders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })

    let response = await request;
    let finalres = await response.json();
    console.log("2nd step res", finalres.id)
    id = finalres.id;

    thirdStep(token, id)
}

async function thirdStep(token, id) {
    let data = {
        "auth_token": token,
        "amount_cents": `${ammount*100}`,
        "expiration": 3600,
        "order_id": id,
        "billing_data": {
            "apartment": "803",
            "email": "claudette09@exa.com",
            "floor": "42",
            "first_name": "Clifford",
            "street": "Ethan Land",
            "building": "8028",
            "phone_number": "+86(8)9135210487",
            "shipping_method": "PKG",
            "postal_code": "01898",
            "city": "Jaskolskiburgh",
            "country": "CR",
            "last_name": "Nicolas",
            "state": "Utah"
        },
        "currency": "EGP",
        "integration_id": 3191514
    }

    let request = fetch("https://accept.paymob.com/api/acceptance/payment_keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })

    let response = await request;
    let finalres = await response.json();
    console.log("3rd step res", finalres.token)
    let tokenThird = finalres.token;
    console.log("token third", tokenThird)

    cardPayment(tokenThird)

}

async function cardPayment(token) {
    console.log("token CP",token)
    let iframeURL = `https://accept.paymob.com/api/acceptance/iframes/711647?payment_token=${token}`;
    location.href = iframeURL;

}

firstStep();
