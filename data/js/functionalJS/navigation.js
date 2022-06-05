function changeClass(obj, classname) 
{
	var ppUl=obj.parentNode.parentNode;
	var pLi=obj.parentNode;
	var lis=ppUl.getElementsByTagName('LI');
/*
	for (var i=3; i<6; i++) 
	{
		var lnk=lis[i].getElementsByTagName('a')[0];
		if (lnk!=obj) 
		{
			lnk.className='';
		}
	}
*/
	obj.className=classname;
}
function hideClasses(theClass) 
{
	var allHTMLTags=document.getElementsByTagName('*');
	for (i=0; i<allHTMLTags.length; i++) 
	{
		if (allHTMLTags[i].className==theClass) 
		{
			allHTMLTags[i].style.display='none';
		}

	}
}
function hideMenu()
{
	document.getElementById('subnavitems_algemeen').style.display="none";
	document.getElementById('subnavitems_zoeken').style.display="none";
	document.getElementById('subnavitems_snel').style.display="none";
}

function showNavs()
{
// weggehaald 19/1/2011 vanwege position fixed in mobiele browsers
/*
		leftnav=document.getElementById('left');
		if (leftnav)
		{
			div = leftnav.getElementsByTagName('div')[0];
			if (div)
				div.style.position='fixed';
		}
*/
		var copyright = jQuery('.copyright');
		if (copyright.html()!=null)
		{
                	jQuery(".copyright").css({display:"block"});
		}
		textc=document.getElementById('text');
		if (textc)
		{
			divs=textc.getElementsByTagName('div');
			for (i=0; i<divs.length;i++)
			{
			//	alert(divs[i].innerHTML);
			
				if (divs[i].innerHTML=='De plaatsen op de kaart zijn aanklikbaar' || divs[i].innerHTML=='De provincies op de kaart zijn aanklikbaar')
				{
					mapdiv=divs[i-1];
					mapdiv.style.position='relative';
				}
			}
		}
		var bread = document.getElementById('breadcrumbs');
			//alert(bread);
		if (bread)
			bread.style.display="block";
}

function swapmenu(menuitem, obj)
{
	if (document.getElementById('subnavitems_' + menuitem).style.display!="none")
	{
		changeClass(obj, '');
		slideup('subnavitems_'+menuitem);
	}
	else
	{
/*
		leftnav=document.getElementById('left');
		if (leftnav)
		{
			div = leftnav.getElementsByTagName('div')[0];
			if (div)
				div.style.position='static';
		}
*/
		textc=document.getElementById('text');
		if (textc)
		{
			divs=textc.getElementsByTagName('div');
			for (i=0; i<divs.length;i++)
			{
			//	alert(divs[i].innerHTML);
				if (divs[i].innerHTML=='De plaatsen op de kaart zijn aanklikbaar' || divs[i].innerHTML=='De provincies op de kaart zijn aanklikbaar')
				{
					mapdiv=divs[i-1];
					mapdiv.style.position='static';
				}
			}
		}
		var copyright = jQuery('.copyright');
		if (copyright.html()!=null)
		{
                	jQuery(".copyright").css({display:"none"});
		}

		var bread = document.getElementById('breadcrumbs');
		if (bread)
			bread.style.display="none";
		changeClass(obj, 'chosen');
		hideMenu();
		slidedown('subnavitems_'+menuitem);
	}
}

