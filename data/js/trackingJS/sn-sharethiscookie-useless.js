
var current_date = new Date;
var cookie_string = "__uset" + "=" + escape("yes");
var expires = new Date (current_date.getTime()+(86400000));
cookie_string += "; expires=" + expires.toGMTString();
cookie_string += "; domain=" + escape (".sharethis.com")+";path=/";
document.cookie = cookie_string;
