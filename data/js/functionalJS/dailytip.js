//To include a page, invoke ajaxinclude(files_array, "ROTATETYPE") in the BODY of page.
//* file_array is the name of the array containing your list of files to include.
//* For "ROTATETYPE", valid values are "dailyw", "dailym", and "random", for each day of the week, each day of the month, and random, respectively.
//* Included file MUST be from the same domain as the page displaying it.

//Enter path to list of files to display.
//For rotatetype="dailyw", there must be 7 files, and for "dailym", 31 files. Otherwise, no restriction:

var includefiles=[
"/SiteCollectionDocuments/DailyTip/1.htm", 
"/SiteCollectionDocuments/DailyTip/2.htm", 
"/SiteCollectionDocuments/DailyTip/3.htm", 
"/SiteCollectionDocuments/DailyTip/4.htm", 
"/SiteCollectionDocuments/DailyTip/5.htm", 
"/SiteCollectionDocuments/DailyTip/6.htm", 
"/SiteCollectionDocuments/DailyTip/7.htm", 
"/SiteCollectionDocuments/DailyTip/8.htm", 
"/SiteCollectionDocuments/DailyTip/9.htm", 
"/SiteCollectionDocuments/DailyTip/10.htm", 
"/SiteCollectionDocuments/DailyTip/11.htm", 
"/SiteCollectionDocuments/DailyTip/12.htm", 
"/SiteCollectionDocuments/DailyTip/13.htm", 
"/SiteCollectionDocuments/DailyTip/14.htm", 
"/SiteCollectionDocuments/DailyTip/15.htm", 
"/SiteCollectionDocuments/DailyTip/16.htm", 
"/SiteCollectionDocuments/DailyTip/17.htm", 
"/SiteCollectionDocuments/DailyTip/18.htm", 
"/SiteCollectionDocuments/DailyTip/19.htm", 
"/SiteCollectionDocuments/DailyTip/20.htm", 
"/SiteCollectionDocuments/DailyTip/21.htm", 
"/SiteCollectionDocuments/DailyTip/22.htm", 
"/SiteCollectionDocuments/DailyTip/23.htm", 
"/SiteCollectionDocuments/DailyTip/24.htm", 
"/SiteCollectionDocuments/DailyTip/25.htm", 
"/SiteCollectionDocuments/DailyTip/26.htm", 
"/SiteCollectionDocuments/DailyTip/27.htm", 
"/SiteCollectionDocuments/DailyTip/28.htm",
"/SiteCollectionDocuments/DailyTip/29.htm",
"/SiteCollectionDocuments/DailyTip/30.htm",
"/SiteCollectionDocuments/DailyTip/31.htm" //Last one, no comma
]

var rootdomain="http://"+window.location.hostname

function ajaxinclude(files_array, rotatetype){
var page_request = false
if (window.XMLHttpRequest) // if Mozilla, Safari etc
page_request = new XMLHttpRequest()
else if (window.ActiveXObject){ // if IE
try {
page_request = new ActiveXObject("Msxml2.XMLHTTP")
} 
catch (e){
try{
page_request = new ActiveXObject("Microsoft.XMLHTTP")
}
catch (e){}
}
}
else
return false
var url=choosefile(files_array, rotatetype)
if (typeof files_array[url]=="undefined"){
document.write("Error: No file for this day has been found.")
return
}
else
url=files_array[url]
page_request.open('GET', url, false) //get page synchronously 
page_request.send(null)
writecontent(page_request)
}

function writecontent(page_request){
if (window.location.href.indexOf("http")==-1 || page_request.status==200)
document.write(page_request.responseText)
}

function choosefile(files_array, rotatetype){
var today=new Date()
var selectedfile=(rotatetype=="dailyw")? today.getDay() : rotatetype=="dailym"? today.getDate() : Math.floor(Math.random()*files_array.length)
if (rotatetype=="dailyw" && selectedfile==0) //if display type=="week days" and today is Sunday 
selectedfile=7
if (rotatetype=="dailyw" || rotatetype=="dailym")
selectedfile--  //remove 1 to sync with array index
return selectedfile
}