import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';

@Component({
  selector: 'app-formulario',
  templateUrl: './formulario.component.html',
  styleUrl: './formulario.component.scss'
})
export class FormularioComponent {
  dataset = {
    age: "",
    sex: "",
    cp: "",
    trestbps: "",
    chol: "",
    fbs: "",
    restecg: "",
    thalach: "",
    exang: "",
    oldpeak: "",
    slope: "",
    ca: "",
    thal: ""
  }
  modelo = { numero1: 0, numero2: 0, numero3: 0 };
  resultado: string = "";

  constructor(private http: HttpClient) { }

  onSubmit() {
    this.http.post<any>('http://localhost:5000/predecir', this.dataset).subscribe(
      data => {
        if (data && data.resultado) {
          this.resultado = data.resultado.toLowerCase();
        } else {
          console.error('Error: No se recibió un resultado válido del servidor.');
        }
      },
      error => {
        console.error('Error al calcular el resultado:', error);
      }
    );
  }
}