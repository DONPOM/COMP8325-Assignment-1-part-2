var timerlen = 4;
var slideAniLen = 300;

var timerID = new Array();
var startTime = new Array();
var obj = new Array();
var endHeight = new Array();
var moving = new Array();
var dir = new Array();

function slidedown(objname){
        if(moving[objname])
	{
                return;
	}

        if(document.getElementById(objname).style.display != "none")
	{
                return; // cannot slide down something that is already visible
	}

        moving[objname] = true;
        dir[objname] = "down";
        startslide(objname);
}

function slideup(objname){
        if(moving[objname])
                return;

        if(document.getElementById(objname).style.display == "none")
	{
                return; // cannot slide up something that is already hidden
	}

        moving[objname] = true;
        dir[objname] = "up";
        startslide(objname);
}

function startslide(objname){
        obj[objname] = document.getElementById(objname);
	obj['subnav'] = document.getElementById('subnav');

        endHeight[objname] = parseInt(obj[objname].style.height);
        startTime[objname] = (new Date()).getTime();

        if(dir[objname] == "down"){
                obj[objname].style.height = "1px";
        }

        obj[objname].style.display = "block";

        timerID[objname] = setInterval('slidetick(\'' + objname + '\');',timerlen);
}

function slidetick(objname){
        var elapsed = (new Date()).getTime() - startTime[objname];

        if (elapsed > slideAniLen)
                endSlide(objname)
        else {
                var d =Math.round(elapsed / slideAniLen * endHeight[objname]);
                if(dir[objname] == "up")
                        d = endHeight[objname] - d;

                obj[objname].style.height = d + "px";
		obj['subnav'].style.height = d+12+"px";
        }

        return;
}

function endSlide(objname)
{
        clearInterval(timerID[objname]);

        if(dir[objname] == "up")
	{
                obj[objname].style.display = "none";
                obj[objname].style.height = "0px";
		obj['subnav'].style.height = "22px";
		showNavs();
	}
	else
	{
                obj[objname].style.height = "350px";
		obj['subnav'].style.height = "382px";
	}

        obj[objname].style.height = endHeight[objname] + "px";

        delete(moving[objname]);
        delete(timerID[objname]);
        delete(startTime[objname]);
        delete(endHeight[objname]);
        delete(obj[objname]);
        delete(dir[objname]);
	
        return;
}
