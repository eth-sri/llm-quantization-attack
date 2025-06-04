function setLanguageOptions() {
    var href = document.location.href;
    var deflt = href.substring(href.indexOf("default=")+8);

    // add deflt as the default option
    document.write("<OPTION value=1>"+