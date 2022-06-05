/*client.js*/
function DisableStateValidator(dropdownlist, staterequired) {
    var validator = (typeof staterequired == 'string' ? document.getElementById(staterequired) : staterequired);
    //alert(staterequired);
    var country = dropdownlist.options[dropdownlist.selectedIndex].value;
    //alert(country);
    var stateRequired = document.getElementById(ClientIDs.star);
    //alert(ClientIDs.star);

    if (country != "United States") {
        stateRequired.innerText = "";
        ValidatorEnable(validator, false);
    }
    else {
        stateRequired.innerText = "*";
        ValidatorEnable(validator, true);

    }
}

function PageQuery(q) {
    if (q.length > 1)
        this.q = q.substring(1, q.length);
    else this.q = null;
    this.keyValuePairs = new Array();
    if (q) {

        for (var i = 0; i < this.q.split("&").length; i++) {
            this.keyValuePairs[i] = this.q.split("&")[i];
        }
    }

    this.getKeyValuePairs = function() { return this.keyValuePairs; };

    this.getValue = function(s) {
        for (var j = 0; j < this.keyValuePairs.length; j++) {
            if (this.keyValuePairs[j].split("=")[0] == s)
                return this.keyValuePairs[j].split("=")[1];
        }
        return false;
    }

    this.getParameters = function() {

        var a = new Array(this.getLength());

        for (var j = 0; j < this.keyValuePairs.length; j++) {
            a[j] = this.keyValuePairs[j].split("=")[0];
        }
        return a;
    };

    this.getLength = function() { return this.keyValuePairs.length; };
}
function queryString(key) {
    var page = new PageQuery(window.location.search);
    return unescape(page.getValue(key));
}

function URLValidate(oSrc, args) {

    args.IsValid = true;
    var url = document.getElementById(ClientIDs.URL).value;
    //var regEx = "(([a-zA-Z][0-9a-zA-Z+\\-\\.]*:)?/{0,2}[0-9a-zA-Z;/?:@&=+$\\.\\-_!~*'()%]+)?(#[0-9a-zA-Z;/?:@&=+$\\.\\-_!~*'()%]+)?";
    //var regEx = "((https?|ftp|gopher|telnet|file|notes|ms-help):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*)";

    //    if(url.substring(0,3) != "www" &&  url.substring(0,3) != "http")
    //    {
    //        args.IsValid = false;
    //    }
    ////    if(!url.match(regEx))
    ////    {
    ////         args.IsValid = false;
    ////    }
    //    else if(ContainsString(url, 'localhost') || ContainsString(url, '127.0.0.1') || ContainsString(url, '192.168'))
    //    {
    //        args.IsValid = false;
    //    } 

    args.IsValid = IsValidURL(url);
}

function WebsiteValidate(oSrc, args) {

    args.IsValid = true;
    var url = document.getElementById(ClientIDs.Website).value;
    //var regEx = "((https?|ftp|gopher|telnet|file|notes|ms-help):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*)";

    //    if(!url.match(regEx))
    //    {
    //         args.IsValid = false;
    //    }
    //    if(url.substring(0,3) != "www" &&  url.substring(0,3) != "http")
    //    {
    //        args.IsValid = false;
    //    }
    //    else if(ContainsString(url, 'localhost') || ContainsString(url, '127.0.0.1'))
    //    {
    //        args.IsValid = false;
    //    }  
    args.IsValid = IsValidURL(url);
}

function IsValidURL(url) {
    var isValid = true;
    if (url.substring(0, 3) != "www" && url.substring(0, 4) != "http") {
        isValid = false;
    }
    else if (ContainsString(url, 'localhost') || ContainsString(url, '127.0.0.1') || ContainsString(url, '192.168.')) {
        isValid = false;
    }

    return isValid;
}


//mouse over
function mouseon(obj) {
    var style = obj.style;
    style.textDecoration = "underline";
}
//mouse off
function mouseoff(obj) {
    var style = obj.style;
    style.textDecoration = "none";
}

//mouse over
function cursor_hand() {
    document.body.style.cursor = 'pointer';
}
//mouse off
function cursor_normal() {
    document.body.style.cursor = 'default';
}


//limit the number of chars in a textbox
function LimitTheCharNumber(textBoxId, noOfChars) {
    //var text = document.getElementById(textBoxId).value;
    var text = textBoxId.value;
    if (text.length > noOfChars) {
        text = text.substr(0, noOfChars);
    }

    //alert(text);

    //document.getElementById(textBoxId).value = text;
    textBoxId.value = text;
}

//check if a country is selected when creating a new user
function CheckCoutrySelected(oSrc, args) {
    args.IsValid = true;
    var country = document.getElementById(ClientIDs.Country);
    var selectedCountry = country.options[country.selectedIndex].value;
    if (selectedCountry.toLowerCase() == '0c') {
        args.IsValid = false;
        //document.getElementById(ClientIDs.CountryRequired).style.visibility =  'visible';
    }
    else {
        //document.getElementById(ClientIDs.CountryRequired).style.visibility =  'none';
    }

}

//in the create user page displays or hiddes the country required message
function SetCountryValidator() {
    //args.IsValid = true;
    var country = document.getElementById(ClientIDs.Country);
    var selectedCountry = country.options[country.selectedIndex].value;
    if (selectedCountry.toLowerCase() == '0c') {
        //args.IsValid = false;
        document.getElementById(ClientIDs.CountryRequired).style.visibility = 'visible';
    }
    else {
        document.getElementById(ClientIDs.CountryRequired).style.visibility = 'hidden';
    }
}

//in the create user page displays or hiddes the terms of use must be checked message
function SetTermsOfUseValidator() {
    //alert('ok');

    if (document.getElementById(ClientIDs.TermsOfUse).checked) {
        document.getElementById(ClientIDs.TermsOfUseChecked).style.display = 'none';
    }
    else {
        document.getElementById(ClientIDs.TermsOfUseChecked).style.display = 'inline';
    }
}


//contact us form validate subject
function MailSubjectValidate(oSrc, args) {
    args.IsVald = true;
    args.IsValid = document.getElementById(ClientIDs.Subject).selectedIndex >= 1;
}

function DisableMailSubjectValidator() {
    if (document.getElementById(ClientIDs.Subject).selectedIndex == 4)
        document.location.href = '/UI/GetPriceOffer.aspx?m=l';

    var validator = document.getElementById(ClientIDs.SubjectSelected);

    if (document.getElementById(ClientIDs.Subject).selectedIndex >= 1) {
        ValidatorEnable(validator, false);
    }
    else {
        ValidatorEnable(validator, true);
    }
}

//apply for trial page: shows or hides the supplementary text field for a product
function ShowSupplementaryInfoTextField(product) {
    var productId = product.getAttribute("value");
    var pname = product.getAttribute("pname");
    var textBoxName = pname + "_textbox";
    var labelName = pname + "_labelx";
    var labelNameReq = pname + "_labelreq";
    var validatorName = "reqSuppInfValidator_" + pname + "_textbox";
    var textboxInf = document.getElementById(ClientIDs[textBoxName]);
    var label = document.getElementById(ClientIDs[labelName]);
    var validator = document.getElementById(ClientIDs[validatorName]);
    var labelReq = document.getElementById(ClientIDs[labelNameReq]);

    if (!product.checked) {
        //alert("hide");
        if (textboxInf != null && label != null && labelReq != null) {
            textboxInf.style.display = 'none';
            label.style.display = 'none';
            labelReq.style.display = 'none';
            ValidatorEnable(validator, false);
            validator.style.display = 'none';
        }

    }
    else {
        //alert("show");
        if (textboxInf != null && label != null && labelReq != null) {
            textboxInf.style.display = 'inline';
            label.style.display = 'inline';
            labelReq.style.display = 'inline';
            ValidatorEnable(validator, true);
            validator.style.display = 'none';
        }
    }

}

//apply for trial page: shows or hides the supplementary text field for a product
function ShowSupplementaryInfoTextFieldForDefault(product, textboxInf, label, validator, labelReq) {
    var productId = product.getAttribute("value");
    var pname = product.getAttribute("pname");


    if (!product.checked) {
        //alert("hide");
        if (textboxInf != null && label != null && labelReq != null && textboxInf != 'undefined'
        && label != 'undefined' && labelReq != 'undefined' && labelReq != null && labelReq != 'undefined') {
            textboxInf.style.display = 'none';
            label.style.display = 'none';
            labelReq.style.display = 'none';
            ValidatorEnable(validator, false);
            validator.style.display = 'none';
        }

    }
    else {
        //alert("show");
        if (textboxInf != null && label != null && labelReq != null && textboxInf != 'undefined'
        && label != 'undefined' && labelReq != 'undefined' && labelReq != null && labelReq != 'undefined') {
            textboxInf.style.display = 'inline';
            label.style.display = 'inline';
            labelReq.style.display = 'inline';
            ValidatorEnable(validator, true);
            validator.style.display = 'none';
        }
    }

}

//create user form
function DisableWebsite(checkbox) {
    var validator = document.getElementById(ClientIDs["WebsiteReq"]);
    var star = document.getElementById(ClientIDs["WebsiteReqLabel"]);

    if (checkbox.checked) {
        //alert("off");   
        ValidatorEnable(validator, false);
        star.style.display = 'none';
        validator.style.display = 'none';
    }
    else {
        //alert("on");
        ValidatorEnable(validator, true);
        star.style.display = 'inline';
        validator.style.display = 'inline';
    }
}

//get price offer form
function DisableWebsitePriceOffer(checkbox) {
    var validator = document.getElementById(ClientIDs["URLRequired"]);
    var star = document.getElementById(ClientIDs["star"]);

    if (checkbox.checked) {
        //alert("off");   
        ValidatorEnable(validator, false);
        star.style.display = 'none';
        validator.style.display = 'none';
    }
    else {
        //alert("on");
        ValidatorEnable(validator, true);
        star.style.display = 'inline';
        validator.style.display = 'inline';
    }
}

//product quicklinks menu
function QuickLinksOnMouseOver(link, tr1, tr2, backGroundImageName) {
    //link.style.backgroundImage = 'url(/App_Themes/Default/quicklinks/'+ backGroundImageName + '.png)';
    if (document.getElementById(ClientIDs[link]) != null) document.getElementById(ClientIDs[link]).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/' + backGroundImageName + ')';
    if (document.getElementById(ClientIDs[tr1]) != null) document.getElementById(ClientIDs[tr1]).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/submenu_bg_selected.png)';
    if (document.getElementById(ClientIDs[tr2]) != null) document.getElementById(ClientIDs[tr2]).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/submenu_bg_selected.png)'
}

function QuickLinksOnMouseOut(link, tr1, tr2, backGroundImageName, menu) {
    var qsMenu = queryString('m');
    //alert(qsMenu);
    if (qsMenu == 'false')
        qsMenu = 'o';
    if (qsMenu != menu)
        if (link != null) link.style.backgroundImage = 'url(/App_Themes/Default/quicklinks/' + backGroundImageName + '.png)'
    if (document.getElementById(ClientIDs[tr1]) != null) document.getElementById(ClientIDs[tr1]).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/submenu_bg.png)';
    if (document.getElementById(ClientIDs[tr2]) != null) document.getElementById(ClientIDs[tr2]).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/submenu_bg.png)';
}


//menu from xml file
function XMLMenuOnMouseOver(link, tr1, tr2, backGroundImageName, menu) {
    //link.style.backgroundImage = 'url(/App_Themes/Default/quicklinks/'+ backGroundImageName + '.png)';
    if (document.getElementById(link) != null) document.getElementById(link).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/' + backGroundImageName + ')';
    if (document.getElementById(tr1) != null) document.getElementById(tr1).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/submenu_bg_selected.png)';
    if (document.getElementById(tr2) != null) document.getElementById(tr2).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/submenu_bg_selected.png)'
    //    if(document.getElementById('selectedMenu')!=null) 
    //    {
    //        if(menu == document.getElementById('selectedMenu').value)
    //            document.getElementById('selectedMenu').value = menu;
    //    }


}

function SetSelectedMenu(menu) {
    if (document.getElementById('selectedMenu') != null) document.getElementById('selectedMenu').value = menu;
}

function XMLMenuOnMouseOut(link, tr1, tr2, backGroundImageName, menu) {
    var qsMenu = queryString('menu');

    if (qsMenu == 'false')
        qsMenu = '';

    if (qsMenu != menu) {
        if (document.getElementById(link) != null) document.getElementById(link).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/' + backGroundImageName + ')';
        if (document.getElementById(tr1) != null) document.getElementById(tr1).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/submenu_bg.png)';
        if (document.getElementById(tr2) != null) document.getElementById(tr2).style.backgroundImage = 'url(/App_Themes/Default/quicklinks/submenu_bg.png)';
    }
}

function isProductSelected(oSrc, args) {
    var div = document.getElementById(ClientIDs.productsPanel).childNodes[0];
    var divContent = div.childNodes[0];
    var checkboxes = divContent.getElementsByTagName('INPUT');


    //alert(tableBody);

    args.IsValid = false;

    for (var i = 0; i < checkboxes.length; i++) {
        var currentTd = checkboxes[i];
        //var listControl = currentTd.childNodes[0];

        if (currentTd.type.toUpperCase() == 'CHECKBOX') {
            if (currentTd.checked == true) {
                //alert('#' + i + ': is checked');
                args.IsValid = true;
                break;
            }
        }
    }
}


function preloadImages() {
    var height = 20;
    var width = 100;
    var path = '/App_Themes/Default/quicklinks/';

    var pictures = new Array(
            "overview.png",
            "overview_selected.png",
            "features.png",
            "features_selected.png",
            "support.png",
            "support_selected.png",
            "downloads.png",
            "downloads_selected.png",
            "licensing.png",
            "licensing_selected.png",
            "gettrial.png",
            "gettrial_selected.png",
            "submenu_bg.png",
            "submenu_bg_selected.png"
        );

    if (document.images) {
        var images = new Array;

        for (var picNum = 0; picNum < pictures.length; picNum++) {
            images[picNum] = new Image(width, height);
            images[picNum].src = (path + pictures[picNum]);
        }
    }

    //alert(images[0].src);

}
//preloadImages();

function ShowText(checkBox, textBox) {
    if (checkBox.checked) {
        //alert('on');
        document.getElementById(textBox).style.display = 'inline';
    }
    else {
        //alert('off');
        document.getElementById(textBox).style.display = 'none';
    }
}


function setLanguage(language, applet) {
    var applet = document.getElementById(applet);
    applet.setLanguage(language);
    //alert(language);
}


function ContainsString(strInitial, strToCheck)
{
    var contains = false;
    var initialLength = strInitial.length;
    var strToCheckLength = strToCheck.length;
    
    for(i = 0; i + strToCheckLength <= initialLength; i++)
    {
        if(strInitial.substr(i, strToCheckLength) == strToCheck)
        {
            contains = true;
            break;
        } 
    }
    return contains;
}

function redirectFramedPage() {

    if (typeof redirect !== 'undefined' && redirect === true) {
        try {
            if (self.location != top.location) {
                if (redirectToNetDania === true) {
                    self.parent.location = "http://www.netdania.com";
                }
                else {
                    self.location = "http://www.netdania.com/netdania.html";
                }
            }
        }
        catch (Exception) { }
    }

}

function calculate_time_zone() {
    var d = new Date();
    var gmtHours = -d.getTimezoneOffset() / 60;

    return gmtHours;  //+ ':0';
}

function countdown_clock(year, month, day, hour, minute, format) {
    //I chose a div as the container for the timer, but
    //it can be an input tag inside a form, or anything
    //who's displayed content can be changed through
    //client-side scripting.
    //html_code = '<div id="countdown" style="border:1px solid #CCCCCC;width:400px; text-align:center" class="counter"></div>';
html_code = '<span id="countdown" style="border:1px solid #CCCCCC;width:400px; text-align:center; padding-left:10px;padding-right:10px" class="counter"></span>';

    document.write(html_code);

    countdown(year, month, day, hour, minute, format);
}

function countdown(year, month, day, hour, minute, format) {
    Today = new Date();
    Todays_Year = Today.getFullYear() - 2000;
    Todays_Month = Today.getMonth();

    //Convert both today's date and the target date into miliseconds.                           
    Todays_Date = (new Date(Todays_Year, Todays_Month, Today.getDate(),
                                 Today.getHours(), Today.getMinutes(), Today.getSeconds())).getTime();
    //Target_Date = (new Date(year, month - 1, day, hour, minute, 00)).getTime();
	Target_Date = (new Date(year, month - 1, day, hour + calculate_time_zone() - 2, minute, 00)).getTime();

    //Find their difference, and convert that into seconds.                  
    Time_Left = Math.round((Target_Date - Todays_Date) / 1000);

    if (Time_Left < 0)
        Time_Left = 0;

    switch (format) {
        case 0:
            //The simplest way to display the time left.
            document.getElementById('countdown').innerHTML = Time_Left + ' seconds';
            break;
        case 1:
            //More datailed.
            days = Math.floor(Time_Left / (60 * 60 * 24));
            Time_Left %= (60 * 60 * 24);
            hours = Math.floor(Time_Left / (60 * 60));
            Time_Left %= (60 * 60);
            minutes = Math.floor(Time_Left / 60);
            Time_Left %= 60;
            seconds = Time_Left;

            dps = 's'; hps = 's'; mps = 's'; sps = 's';
            //ps is short for plural suffix.
            if (days == 1) dps = '';
            if (hours == 1) hps = '';
            if (minutes == 1) mps = '';
            if (seconds == 1) sps = '';

            document.getElementById('countdown').innerHTML ='' + days + ' day' + dps + ' ';
            document.getElementById('countdown').innerHTML += hours + ' hour' + hps + ' ';
            document.getElementById('countdown').innerHTML += minutes + ' minute' + mps +' and ';
            document.getElementById('countdown').innerHTML += '<span style="color:#D92A2A">' + seconds + '</span>' + ' second' + sps;
            break;
        default:
            document.getElementById('countdown').innerHTML = Time_Left + ' seconds';
    }

    //Recursive call, keeps the clock ticking.
    setTimeout('countdown(' + year + ',' + month + ',' + day + ',' + hour + ',' + minute + ',' + format + ');', 1000);
};
/*client.js end*/
