function disp_confirm(vnm){
	var name=confirm("Are you sure you want to deactivate \n" + vnm + "?")
}

function disp_delete(vid,vnm){
	var name=confirm("Are you sure you want to delete \n" + vid + " - " + vnm + "?")
}

function mOver(id) {
	getElementById(id).style.display = 'block';
}

function mOut(id) {
	getElementById(id).style.display = 'none';
}

function getElementById(id) {
	var element;
	if (document.getElementById) element = document.getElementById(id);
	else if (document.all) element = document.all[id];
	else if (document.layers) element = document.layers[id];
	return element;
}

function SetFlashSize(myDivVar,myHeight){
	var whichDiv = document.getElementById(myDivVar);
	if (whichDiv){
		document.getElementById(myDivVar).style.height=myHeight+'px';
		}
}

function display(url, name, r, s, t, w, h) {
	mediumWin = window.open(url, name,'resizable=' + r+ ',scrollbars=' + s + ',toolbar=' + t + ',status=0,width=' + w + ',height=' + h);
	mediumWin.focus();
}