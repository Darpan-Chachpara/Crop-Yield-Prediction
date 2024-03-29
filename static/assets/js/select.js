var a = {
	'Arhar/Tur': ['Kharif'],
    'Bajra': ['Kharif'],
    'Gram': ['Kharif', 'Rabi'],
    'Jowar': ['Kharif', 'Rabi'],
    'Maize': ['Kharif', 'Rabi', 'Summer'],
    'Moong(Green Gram)': ['Kharif'],
    'Pulses total': ['Kharif'],
    'Ragi': ['Kharif'],
    'Rice': ['Kharif', 'Summer'],
    'Sugarcane': ['Kharif', 'Whole Year'],
    'Total foodgrain': ['Kharif'],
    'Urad': ['Kharif'],
    'Other Rabi pulses': ['Rabi'],
    'Wheat': ['Rabi'],
    'Cotton(lint)': ['Whole Year', 'Kharif'],
    'Castor seed': ['Kharif'],
    'Groundnut': ['Kharif', 'Summer'],
    'Niger seed': ['Kharif'],
    'Other Cereals & Millets': ['Kharif', 'Rabi'],
    'Other Kharif pulses': ['Kharif'],
    'Sesamum': ['Kharif', 'Rabi'],
    'Soyabean': ['Kharif'],
    'Sunflower': ['Kharif', 'Rabi', 'Summer'],
    'Linseed': ['Rabi'],
    'Safflower': ['Rabi'],
    'Small millets': ['Kharif', 'Rabi'],
    'Rapeseed &Mustard': ['Rabi']
}

var b = {
'AHMEDNAGAR': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Ragi', 'Rice', 'Sugarcane', 'Total foodgrain', 'Urad',
 'Other Rabi pulses', 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut',
 'Niger seed', 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum',
 'Soyabean', 'Sunflower', 'Linseed', 'Safflower', 'Small millets',
 'Rapeseed &Mustard'],
'AKOLA': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Groundnut', 'Other Cereals & Millets',
 'Other Kharif pulses', 'Sesamum', 'Soyabean', 'Sugarcane', 'Sunflower',
 'Safflower', 'Small millets', 'Rapeseed &Mustard'],
'AMRAVATI': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut', 'Niger seed',
 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum', 'Soyabean',
 'Sugarcane', 'Sunflower', 'Linseed', 'Rapeseed &Mustard', 'Safflower',
 'Small millets'],
'AURANGABAD': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Wheat', 'Cotton(lint)',
 'Groundnut', 'Niger seed', 'Other Cereals & Millets', 'Other Kharif pulses',
 'Sesamum', 'Soyabean', 'Sugarcane', 'Sunflower', 'Linseed',
 'Rapeseed &Mustard', 'Safflower', 'Castor seed', 'Small millets'],
'BEED': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut', 'Niger seed',
 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum', 'Soyabean',
 'Sugarcane', 'Sunflower', 'Linseed', 'Rapeseed &Mustard', 'Safflower',
 'Small millets'],
'BHANDARA': ['Arhar/Tur', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)', 'Pulses total',
 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses', 'Wheat', 'Castor seed',
 'Groundnut', 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum',
 'Soyabean', 'Sugarcane', 'Linseed', 'Rapeseed &Mustard', 'Safflower',
 'Small millets'],
'BULDHANA': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Wheat', 'Cotton(lint)',
 'Castor seed', 'Groundnut', 'Niger seed', 'Other Cereals & Millets',
 'Other Kharif pulses', 'Sesamum', 'Soyabean', 'Sugarcane', 'Sunflower',
 'Linseed', 'Other Rabi pulses', 'Rapeseed &Mustard', 'Safflower',
 'Small millets'],
'CHANDRAPUR': ['Arhar/Tur', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)', 'Pulses total',
 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses', 'Wheat',
 'Cotton(lint)', 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum',
 'Soyabean', 'Linseed', 'Rapeseed &Mustard', 'Safflower', 'Sunflower',
 'Groundnut', 'Castor seed', 'Niger seed', 'Small millets'],
'DHULE': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Ragi', 'Rice', 'Sugarcane', 'Total foodgrain', 'Urad',
 'Other Rabi pulses', 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut',
 'Niger seed', 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum',
 'Soyabean', 'Sunflower', 'Rapeseed &Mustard', 'Safflower', 'Small millets'],
'GADCHIROLI': ['Arhar/Tur', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)', 'Pulses total',
 'Rice', 'Total foodgrain', 'Urad', 'Wheat', 'Other Cereals & Millets',
 'Other Kharif pulses', 'Sesamum', 'Soyabean', 'Linseed', 'Other Rabi pulses',
 'Rapeseed &Mustard', 'Safflower', 'Sunflower', 'Groundnut', 'Small millets',
 'Cotton(lint)'],
'GONDIA': ['Arhar/Tur', 'Bajra', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Other Kharif pulses', 'Rice', 'Sesamum', 'Soyabean', 'Urad', 'Gram', 'Linseed',
 'Wheat', 'Groundnut', 'Castor seed', 'Small millets', 'Other Rabi pulses',
 'Rapeseed &Mustard', 'Sugarcane'],
'HINGOLI': ['Arhar/Tur', 'Cotton(lint)', 'Groundnut', 'Niger seed', 'Sesamum', 'Soyabean',
 'Sunflower', 'Urad', 'Linseed', 'Safflower', 'Sugarcane', 'Bajra',
 'Castor seed', 'Jowar', 'Maize', 'Moong(Green Gram)', 'Other Kharif pulses',
 'Rice', 'Small millets', 'Gram', 'Other Rabi pulses', 'Rapeseed &Mustard',
 'Wheat'],
'JALGAON': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Sugarcane', 'Total foodgrain', 'Urad', 'Wheat',
 'Cotton(lint)', 'Castor seed', 'Groundnut', 'Other Cereals & Millets',
 'Other Kharif pulses', 'Sesamum', 'Soyabean', 'Sunflower',
 'Other Rabi pulses', 'Safflower', 'Small millets', 'Rapeseed &Mustard'],
'JALNA': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut', 'Niger seed',
 'Other Kharif pulses', 'Sesamum', 'Soyabean', 'Sugarcane', 'Sunflower',
 'Linseed', 'Safflower', 'Rapeseed &Mustard', 'Small millets'],
'KOLHAPUR': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Ragi', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Groundnut', 'Niger seed', 'Other Cereals & Millets',
 'Other Kharif pulses', 'Soyabean', 'Sugarcane', 'Sunflower', 'Castor seed',
 'Small millets'],
'LATUR': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Groundnut', 'Niger seed', 'Other Cereals & Millets',
 'Other Kharif pulses', 'Sesamum', 'Soyabean', 'Sugarcane', 'Sunflower',
 'Linseed', 'Rapeseed &Mustard', 'Safflower', 'Castor seed', 'Small millets'],
'NAGPUR': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut',
 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum', 'Soyabean',
 'Sugarcane', 'Sunflower', 'Linseed', 'Rapeseed &Mustard', 'Niger seed',
 'Small millets'],
'NANDED': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Moong(Green Gram)', 'Pulses total',
 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses', 'Wheat', 'Maize',
 'Cotton(lint)', 'Castor seed', 'Groundnut', 'Niger seed',
 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum', 'Soyabean',
 'Sugarcane', 'Sunflower', 'Linseed', 'Rapeseed &Mustard', 'Safflower',
 'Small millets'],
'NANDURBAR': ['Arhar/Tur', 'Cotton(lint)', 'Groundnut', 'Maize', 'Moong(Green Gram)',
 'Niger seed', 'Other Kharif pulses', 'Rice', 'Sesamum', 'Soyabean',
 'Sunflower', 'Urad', 'Gram', 'Other Rabi pulses', 'Safflower', 'Small millets',
 'Wheat', 'Sugarcane', 'Bajra', 'Castor seed', 'Jowar', 'Ragi',
 'Rapeseed &Mustard'],
'NASHIK': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Ragi', 'Rice', 'Sugarcane', 'Total foodgrain', 'Urad',
 'Other Rabi pulses', 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut',
 'Niger seed', 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum',
 'Soyabean', 'Sunflower', 'Rapeseed &Mustard', 'Safflower', 'Small millets',
 'Linseed'],
'OSMANABAD': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut', 'Niger seed',
 'Other Kharif pulses', 'Sesamum', 'Soyabean', 'Sugarcane', 'Sunflower',
 'Linseed', 'Other Cereals & Millets', 'Safflower', 'Small millets',
 'Rapeseed &Mustard'],
'PARBHANI': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut', 'Niger seed',
 'Other Kharif pulses', 'Sesamum', 'Soyabean', 'Sugarcane', 'Sunflower',
 'Linseed', 'Other Cereals & Millets', 'Rapeseed &Mustard', 'Safflower',
 'Small millets'],
'PUNE': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Ragi', 'Rice', 'Sugarcane', 'Total foodgrain', 'Urad',
 'Other Rabi pulses', 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut',
 'Niger seed', 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum',
 'Soyabean', 'Sunflower', 'Linseed', 'Safflower', 'Small millets',
 'Rapeseed &Mustard'],
'SANGLI': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Ragi', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Groundnut', 'Niger seed', 'Other Kharif pulses',
 'Sesamum', 'Soyabean', 'Sugarcane', 'Sunflower', 'Other Cereals & Millets',
 'Rapeseed &Mustard', 'Safflower', 'Small millets'],
'SATARA': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Ragi', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Groundnut', 'Niger seed', 'Other Cereals & Millets',
 'Other Kharif pulses', 'Sesamum', 'Soyabean', 'Sugarcane', 'Sunflower',
 'Linseed', 'Rapeseed &Mustard', 'Safflower', 'Castor seed', 'Small millets'],
'SOLAPUR': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Sugarcane', 'Total foodgrain', 'Urad',
 'Other Rabi pulses', 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut',
 'Niger seed', 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum',
 'Soyabean', 'Sunflower', 'Linseed', 'Safflower', 'Small millets',
 'Rapeseed &Mustard'],
'WARDHA': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Groundnut', 'Other Cereals & Millets',
 'Other Kharif pulses', 'Sesamum', 'Soyabean', 'Sugarcane', 'Linseed',
 'Rapeseed &Mustard', 'Safflower', 'Sunflower', 'Castor seed', 'Small millets'],
'WASHIM': ['Arhar/Tur', 'Bajra', 'Cotton(lint)', 'Groundnut', 'Jowar', 'Maize',
 'Moong(Green Gram)', 'Niger seed', 'Other Kharif pulses', 'Rice', 'Sesamum',
 'Soyabean', 'Sunflower', 'Urad', 'Gram', 'Other Rabi pulses', 'Safflower',
 'Wheat', 'Sugarcane', 'Castor seed', 'Small millets'],
'YAVATMAL': ['Arhar/Tur', 'Bajra', 'Gram', 'Jowar', 'Maize', 'Moong(Green Gram)',
 'Pulses total', 'Rice', 'Total foodgrain', 'Urad', 'Other Rabi pulses',
 'Wheat', 'Cotton(lint)', 'Castor seed', 'Groundnut',
 'Other Cereals & Millets', 'Other Kharif pulses', 'Sesamum', 'Soyabean',
 'Sugarcane', 'Sunflower', 'Linseed', 'Safflower', 'Niger seed',
 'Small millets', 'Rapeseed &Mustard']
}

function selCrop() {
	var crop = document.getElementById("cr").value;
    console.log(crop);

    str = "";
    a[crop].forEach(function(c) {
        str = str + "<option>" + c + "</option>";
    });
    document.getElementById("se").innerHTML = str;
}

function selDist() {
    var dist = document.getElementById("di").value;
    console.log(dist);

    str = "";
    b[dist].forEach(function(c) {
        str = str + "<option>" + c + "</option>";
    });
    document.getElementById("cr").innerHTML = str;
}

if(window.location.pathname == "/") {
    window.onload = selCrop();
    window.onload = selDist();
}