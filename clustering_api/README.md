# Newstuck-ArticleClusterinng

- Text classification and representation Models Used For Tamil and English is FastText By Facebook
- Language Detection is Done By langdetect 1.0.8 (Python Package)

## Contains Two Parts 
 - Language Detection
- Cluster the Similar Titles



## Cluster Similar Titles By Followed Steps
    - Get the Titles and Language from the Input
    - Based on the language Load the Model and generate the Sentence Vector
    - Finding the Optimal No of Clusters
    - By using the Optimal Cluster Value and K-Means Algorithm Cluster the Titles.

## The Exposed Api are
    http://newscluster.southindia.azurecontainer.io/api/v1/detectlanguage
    http://newscluster.southindia.azurecontainer.io/api/v1/getcluster

    * HTTP Post Method
    
## Example input Data : For Get Cluster
``` 
{
    "Language" : "ta",
    "Titles" : [
    {
        "Title": "அடல் சுரங்கப்பாதை: இமாசல மக்களுக்கு வரப்பிரசாதம்: மத்திய அமைச்சர் அமித்ஷா கருத்து",
    },
    {
        "Title": "அம்பானி, அதானி விவசாயிகளின் நிலம், பயிர்களை குறைந்த விலைக்கு வாங்குவதை மோடி விரும்புகிறார்… ராகுல் காந்தி",
    },
    {
        "Title": "நடிகைக்காக நீதி கேட்டவர்கள் ஹத்ராஸ் சம்பவத்துக்கு பிறகு அமைதியாகி விட்டார்கள்…. சஞ்சய் ரவுத்",
    },
    {
        "Title": "இஸ்ரேலில் கொரோனா கட்டுப்பாடுகளுக்கு எதிராக கொதித்தெழுந்த மக்கள் - வீதிகளில் இறங்கி ஆவேச போராட்டம்",
    },
    {
        "Title": "சொந்த பலத்தில் குமாரசாமியால் ஆட்சி அமைக்க முடியாது - சித்தராமையா",
    },
    {
        "Title": "‘கோவேக்சின்’ தடுப்பூசி 3-வது கட்ட பரிசோதனை அடுத்த வாரம் தொடங்கும் - எஸ்.ஆர்.எம். மருத்துவக்கல்லூரி, ஆராய்ச்சி நிலைய டீன் சுந்தரம் தகவல்"
    },
    {
        "Title": "உத்தரபிரதேச சம்பவத்திற்கு நீதி கேட்டு கனிமொழி தலைமையில் பேரணி - மு.க.ஸ்டாலின் அறிவிப்பு",
    },
    {
        "Title": "கவர்னரை இன்று சந்திக்கிறார் எடப்பாடி பழனிசாமி - கொரோனா தடுப்பு நடவடிக்கைகள் குறித்து விளக்கம்",
    }]
}

```

## Example input Data : For Language Detection
```
 [
	{
		"Title" : "Dream Warrior Pictures starts its shooting again"
	},
	{
		"Title" : "Osaka Tamil International Film Festival News"
	},
	{
		"Title" : "BJP leader shot dead in West Bengal, Vijayvargiya demands CBI inquiry"
	},
	{
		"Title" : "Prolific voice of Indian melodies"
	},
	{
		"Title" : "அடல் சுரங்கப்பாதை: இமாசல மக்களுக்கு வரப்பிரசாதம்: மத்திய அமைச்சர் அமித்ஷா கருத்து"
	},
	{
		"Title" : "அம்பானி, அதானி விவசாயிகளின் நிலம், பயிர்களை குறைந்த விலைக்கு வாங்குவதை மோடி விரும்புகிறார்… ராகுல் காந்தி"
	},
	{
		"Title" : "நடிகைக்காக நீதி கேட்டவர்கள் ஹத்ராஸ் சம்பவத்துக்கு பிறகு அமைதியாகி விட்டார்கள்…. சஞ்சய் ரவுத்"
	}
]
```
