import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { FormularioComponent } from './component/formulario/formulario.component';
import { FooterComponent } from './component/footer/footer.component';

const routes: Routes = [
  {path: 'formulario', component: FooterComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
