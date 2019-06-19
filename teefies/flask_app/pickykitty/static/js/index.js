// get the race list for text suggestion
sendRequest({ url: '/racelist', method: 'GET' })
  .then(json => {
    const races = json.data

    // update global variable racelist
    // swap key/values for fast lookup later
    raceslist = Object.keys(races)
      .reduce((obj, key) => ({ ...obj, [races[key]]: key }), {})

    autocomplete(document.getElementById("race-input"), Object.values(races))
  })