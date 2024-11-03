from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        descripteur = request.form.get('descripteur')
        distance = request.form.get('distance')
        corpus = request.form.get('corpus')
        phrase = request.form.get('phrase')
        k = int(request.form.get('k'))

        # Récupération des options cochées
        n_option = request.form.get('n_option')  # Valeur de N1, N2, ou aucune
        freq_option = request.form.get('freq_option')  # Valeur de Bin, Freq, ou Freq_total

        # Logique pour traiter les entrées et générer des résultats
        phrases = [f"Phrase {i + 1}: Résultat pour '{phrase}' avec N: {n_option} et Freq: {freq_option}" for i in range(k)]

        return render_template('index.html', phrases=phrases)

    return render_template('index.html', phrases=None)

if __name__ == '__main__':
    app.run(debug=True)